"""
Repositories — couche d'accès aux données
Principe SOLID D : les services dépendent d'abstractions (Protocol), pas de SQLAlchemy directement
"""
from typing import Protocol, Optional
from uuid import UUID
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.models.models import User, CompetenceProfile, Session, Message, KnowledgeChunk


# ── Protocols (interfaces) ────────────────────────────────────────────────────

class IUserRepository(Protocol):
    async def get_by_email(self, email: str) -> Optional[User]: ...
    async def get_by_id(self, user_id: UUID) -> Optional[User]: ...
    async def create(self, user: User) -> User: ...


class ISessionRepository(Protocol):
    async def create(self, session: Session) -> Session: ...
    async def get_by_id(self, session_id: UUID) -> Optional[Session]: ...
    async def update_tokens(self, session_id: UUID, tokens: int) -> None: ...
    async def update_summary(self, session_id: UUID, summary: str) -> None: ...


class ICompetenceRepository(Protocol):
    async def get_by_user(self, user_id: UUID) -> list[CompetenceProfile]: ...
    async def upsert(self, user_id: UUID, topic: str, score: float) -> CompetenceProfile: ...


# ── Implementations ───────────────────────────────────────────────────────────

class UserRepository:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self._db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        result = await self._db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def create(self, user: User) -> User:
        self._db.add(user)
        await self._db.flush()
        return user


class SessionRepository:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def create(self, session: Session) -> Session:
        self._db.add(session)
        await self._db.flush()
        return session

    async def get_by_id(self, session_id: UUID) -> Optional[Session]:
        result = await self._db.execute(
            select(Session)
            .where(Session.id == session_id)
            .options(selectinload(Session.messages), selectinload(Session.user))
        )
        return result.scalar_one_or_none()

    async def add_message(self, message: Message) -> Message:
        self._db.add(message)
        await self._db.flush()
        return message

    async def update_tokens(self, session_id: UUID, additional_tokens: int) -> None:
        await self._db.execute(
            update(Session)
            .where(Session.id == session_id)
            .values(total_tokens=Session.total_tokens + additional_tokens)
        )

    async def update_summary(self, session_id: UUID, summary: str) -> None:
        await self._db.execute(
            update(Session).where(Session.id == session_id).values(summary=summary)
        )


class CompetenceRepository:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def get_by_user(self, user_id: UUID) -> list[CompetenceProfile]:
        result = await self._db.execute(
            select(CompetenceProfile).where(CompetenceProfile.user_id == user_id)
        )
        return list(result.scalars().all())

    async def upsert(self, user_id: UUID, topic: str, score: float) -> CompetenceProfile:
        result = await self._db.execute(
            select(CompetenceProfile)
            .where(CompetenceProfile.user_id == user_id, CompetenceProfile.topic == topic)
        )
        profile = result.scalar_one_or_none()
        if profile:
            profile.score = score
            # SM-2 simple : augmenter l'intervalle si score > 70
            if score > 70:
                profile.review_interval_days = min(profile.review_interval_days * 2, 30)
            else:
                profile.review_interval_days = 1
        else:
            profile = CompetenceProfile(user_id=user_id, topic=topic, score=score)
            self._db.add(profile)
        await self._db.flush()
        return profile
