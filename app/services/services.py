"""
Services — couche métier
SOLID S : chaque service = une responsabilité
SOLID D : dépend des interfaces repository
"""
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.maia_agent import maia_agent, build_system_prompt
from app.models.models import User, Session, Message, MessageRoleEnum, SessionModeEnum, DiagnosticSession
from app.rag.rag_service import RAGService
from app.repositories.repositories import (
    UserRepository, SessionRepository, CompetenceRepository, DiagnosticSessionRepository
)
from app.schemas.schemas import (
    RegisterRequest, TopicScore, CompetenceResponse
)
from app.core.security import hash_password, verify_password, create_access_token, create_refresh_token

logger = logging.getLogger(__name__)

class AuthService:
    """Responsabilité : authentification et gestion des tokens JWT"""

    def __init__(self, db: AsyncSession):
        self._repo = UserRepository(db)

    async def register(self, req: RegisterRequest) -> tuple[User, str, str]:
        existing = await self._repo.get_by_email(req.email)
        if existing:
            raise ValueError("Un compte existe déjà avec cet email")

        user = User(
            email=req.email,
            hashed_password=hash_password(req.password),
            name=req.name or req.email.split("@")[0],
            vertical=req.vertical,
            exam_date=req.exam_date,
        )
        user = await self._repo.create(user)
        return user, create_access_token(str(user.id)), create_refresh_token(str(user.id))

    async def login(self, email: str, password: str) -> tuple[str, str]:
        user = await self._repo.get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            raise ValueError("Email ou mot de passe incorrect")
        return create_access_token(str(user.id)), create_refresh_token(str(user.id))


class DiagnosticService:
    """Responsabilité : génération et évaluation du diagnostic initial"""

    def __init__(self, db: AsyncSession):
        self._user_repo = UserRepository(db)
        self._comp_repo = CompetenceRepository(db)
        self._diag_repo = DiagnosticSessionRepository(db)

    async def start(self, user_id: UUID) -> dict:
        user = await self._user_repo.get_by_id(user_id)
        if not user:
            raise ValueError("Utilisateur introuvable")

        questions = await maia_agent.generate_diagnostic_questions(
            vertical=user.vertical.value,
            user_name=user.name or "Candidat",
        )

        diag = DiagnosticSession(
            user_id=user_id,
            questions=questions,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=6),
        )
        diag = await self._diag_repo.create(diag)

        return {"diagnostic_id": str(diag.id), "questions": questions}

    async def submit(self, user_id: UUID, diagnostic_id: str, answers: list[dict]) -> dict:
        try:
            diag_uuid = UUID(diagnostic_id)
        except Exception:
            raise ValueError("Session de diagnostic introuvable ou expirée")

        diag = await self._diag_repo.get_active_by_id(diag_uuid, user_id)
        if not diag:
            raise ValueError("Session de diagnostic introuvable ou expirée")

        questions = diag.questions or []

        evaluation = await maia_agent.evaluate_diagnostic_answers(questions, answers)

        # Mettre à jour le profil de compétences
        scores = evaluation.get("scores", {})
        topic_scores = []
        for topic, score in scores.items():
            try:
                numeric_score = float(score)
            except Exception:
                numeric_score = 0.0

            await self._comp_repo.upsert(user_id, topic, numeric_score)
            status = "maîtrisé" if numeric_score >= 70 else ("moyen" if numeric_score >= 40 else "lacune")
            topic_scores.append(TopicScore(topic=topic, score=numeric_score, status=status))

        # Marquer le diagnostic comme consommé (idempotence)
        await self._diag_repo.consume(diag_uuid, user_id)

        strong = [t.topic for t in topic_scores if t.score >= 70]
        weak = [t.topic for t in topic_scores if t.score < 40]

        return {
            "scores": topic_scores,
            "summary": evaluation.get("summary", ""),
            "strong_topics": strong,
            "weak_topics": weak,
        }


class SessionService:
    """Responsabilité : gestion du cycle de vie des sessions pédagogiques"""

    def __init__(self, db: AsyncSession):
        self._db = db
        self._session_repo = SessionRepository(db)
        self._user_repo = UserRepository(db)
        self._comp_repo = CompetenceRepository(db)
        self._rag = RAGService(db)

    async def start_session(self, user_id: UUID, mode: SessionModeEnum, topic: str | None) -> Session:
        session = Session(
            user_id=user_id,
            mode=mode,
            topic=topic,
        )
        return await self._session_repo.create(session)

    async def stream_message(
        self,
        session_id: UUID,
        user_id: UUID,
        content: str,
    ) -> AsyncIterator[str]:
        """
        Traitement d'un message utilisateur avec streaming SSE
        Pipeline : RAG search → build prompt → stream LLM → save message
        """
        session = await self._session_repo.get_by_id(session_id)
        if not session or str(session.user_id) != str(user_id):
            raise ValueError("Session introuvable ou non autorisée")

        user = await self._user_repo.get_by_id(user_id)
        competences = await self._comp_repo.get_by_user(user_id)

        # 1. Recherche RAG
        rag_chunks = await self._rag.search(
            query=content,
            top_k=3,
            min_similarity=0.7,
            topic_filter=session.topic,
        )
        rag_context = self._rag.format_for_prompt(rag_chunks)

        # 2. Profil de compétences
        strong = [c.topic for c in competences if c.score >= 70]
        weak = [c.topic for c in competences if c.score < 40]

        # 3. Compression si historique long (> 2000 tokens)
        existing_messages = session.messages or []
        total_chars = sum(len(m.content) for m in existing_messages)
        summary = session.summary

        if total_chars > 8000 and not summary:  # ~2000 tokens
            logger.info(f"Session {session_id} : compression du contexte ({total_chars} chars)")
            history_for_compression = [
                {"role": m.role.value, "content": m.content} for m in existing_messages
            ]
            summary = await maia_agent.compress_session(history_for_compression)
            await self._session_repo.update_summary(session_id, summary)

        # 4. Build system prompt (3 couches)
        system_prompt = build_system_prompt(
            user_name=user.name or "Candidat",
            vertical=user.vertical.value,
            exam_date=user.exam_date,
            strong_topics=strong,
            weak_topics=weak,
            session_mode=session.mode.value,
            session_topic=session.topic,
            previous_summary=summary,
            rag_context=rag_context,
        )

        # 5. Historique de conversation (fenêtre glissante)
        conversation_history = [
            {"role": m.role.value, "content": m.content}
            for m in existing_messages[-20:]
        ]

        # 6. Sauvegarder le message utilisateur
        user_message = Message(
            session_id=session_id,
            role=MessageRoleEnum.user,
            content=content,
            tokens_used=len(content.split()),  # approximation
        )
        self._db.add(user_message)
        await self._db.flush()

        # 7. Stream LLM et accumuler la réponse
        full_response = ""
        async for token in maia_agent.stream_response(content, conversation_history, system_prompt):
            full_response += token
            yield token

        # 8. Sauvegarder la réponse assistant
        estimated_tokens = len(full_response.split())
        assistant_message = Message(
            session_id=session_id,
            role=MessageRoleEnum.assistant,
            content=full_response,
            tokens_used=estimated_tokens,
        )
        self._db.add(assistant_message)
        await self._session_repo.update_tokens(session_id, estimated_tokens)
        await self._db.flush()

        logger.info(f"Session {session_id} : message traité, ~{estimated_tokens} tokens")

    async def get_history(self, session_id: UUID, user_id: UUID) -> dict:
        session = await self._session_repo.get_by_id(session_id)
        if not session or str(session.user_id) != str(user_id):
            raise ValueError("Session introuvable ou non autorisée")
        return {
            "session_id": session_id,
            "messages": session.messages,
            "total_tokens": session.total_tokens,
        }


class ProfileService:
    """Responsabilité : consultation du profil de compétences"""

    def __init__(self, db: AsyncSession):
        self._comp_repo = CompetenceRepository(db)
        self._user_repo = UserRepository(db)

    async def get_competences(self, user_id: UUID) -> dict:
        user = await self._user_repo.get_by_id(user_id)
        competences = await self._comp_repo.get_by_user(user_id)

        comp_responses = []
        for c in competences:
            status = "maîtrisé" if c.score >= 70 else ("moyen" if c.score >= 40 else "lacune")
            comp_responses.append(CompetenceResponse(
                topic=c.topic,
                score=c.score,
                status=status,
                review_interval_days=c.review_interval_days,
                last_updated=c.last_updated,
            ))

        days_until_exam = None
        if user and user.exam_date:
            delta = user.exam_date.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)
            days_until_exam = max(0, delta.days)

        return {"competences": comp_responses, "days_until_exam": days_until_exam}
