"""
Agent MAÏA — Orchestration LLM Groq
Architecture SOLID :
  S — responsabilité unique : orchestration LLM + prompts
  O — extensible : nouveaux verticaux sans modifier le code
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional
from groq import AsyncGroq
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Cache in-memory (clé = hash du prompt normalisé → réponse)
_prompt_cache: dict[str, str] = {}

# ── Vertical configs ──────────────────────────────────────────────────────────

VERTICAL_CONFIGS = {
    "concours": {
        "name": "Gardien de la Paix",
        "programme_summary": "Droit pénal, procédure pénale, institutions policières, hiérarchie des grades, droits des citoyens, code de déontologie",
        "epreuves": "Culture générale, raisonnement logique, compréhension de texte, entretien oral",
    },
    "bac": {
        "name": "Baccalauréat",
        "programme_summary": "Programme officiel du baccalauréat général ou technologique",
        "epreuves": "Épreuves écrites par matière, grand oral",
    },
    "prepa": {
        "name": "Classes Préparatoires",
        "programme_summary": "Programme des classes préparatoires aux grandes écoles",
        "epreuves": "Concours d'entrée aux grandes écoles",
    },
}


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_system_prompt(
    user_name: str,
    vertical: str,
    exam_date: Optional[datetime],
    strong_topics: list[str],
    weak_topics: list[str],
    session_mode: str,
    session_topic: Optional[str],
    previous_summary: Optional[str],
    rag_context: str,
) -> str:
    """
    Construit le system prompt en 3 couches selon le cahier des charges
    """
    vc = VERTICAL_CONFIGS.get(vertical, VERTICAL_CONFIGS["concours"])
    days_remaining = ""
    if exam_date:
        delta = exam_date.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)
        days_remaining = f" ({max(0, delta.days)} jours)"

    # Couche 1 — Persona globale (fixe)
    layer1 = """Tu es Maïa, une professeure IA bienveillante et exigeante. Tu utilises la méthode Feynman : tu expliques simplement, tu poses des questions pour vérifier la compréhension, tu ne donnes JAMAIS la réponse directe à un exercice avant que l'utilisateur ait essayé. Si l'utilisateur te demande directement la réponse, tu poses une question socratique à la place."""

    # Couche 2 — Contexte vertical (paramétrique)
    layer2 = f"""Tu prépares {user_name} au concours de {vc['name']}.
Programme officiel : {vc['programme_summary']}
Date de l'examen : {exam_date.strftime('%d/%m/%Y') if exam_date else 'non définie'}{days_remaining}.
Épreuves : {vc['epreuves']}"""

    # Couche 3 — Contexte utilisateur (dynamique)
    layer3_parts = [
        "Profil de compétences :",
        f"- Points forts : {', '.join(strong_topics) if strong_topics else 'non encore évalués'}",
        f"- Lacunes identifiées : {', '.join(weak_topics) if weak_topics else 'aucune identifiée'}",
        f"- Session actuelle : mode={session_mode}, topic={session_topic or 'général'}",
    ]
    if previous_summary:
        layer3_parts.append(f"- Résumé session précédente : {previous_summary}")
    if rag_context:
        layer3_parts.append(f"\n{rag_context}")

    layer3 = "\n".join(layer3_parts)

    return f"{layer1}\n\n---\n\n{layer2}\n\n---\n\n{layer3}"


def _cache_key(prompt: str) -> str:
    normalized = " ".join(prompt.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


# ── Groq async client (streaming) ────────────────────────────────────────────

class MaiaAgent:
    """
    Agent MAÏA — wraps Groq API avec streaming SSE
    """

    def __init__(self):
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    async def stream_response(
        self,
        user_message: str,
        conversation_history: list[dict],
        system_prompt: str,
    ) -> AsyncIterator[str]:
        """
        Stream la réponse du LLM token par token (SSE)
        """
        # Vérifier le cache d'abord
        full_prompt = system_prompt + user_message
        cache_key = _cache_key(full_prompt)
        if cache_key in _prompt_cache:
            logger.info(f"Cache hit pour le prompt (hash={cache_key[:8]}...)")
            yield _prompt_cache[cache_key]
            return

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history[-20:])  # Fenêtre glissante de 20 messages
        messages.append({"role": "user", "content": user_message})

        full_response = ""
        total_tokens = 0

        stream = await self._client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                yield delta

        # Mettre en cache
        _prompt_cache[cache_key] = full_response
        logger.info(f"Réponse générée et mise en cache (hash={cache_key[:8]}...)")

    async def generate_diagnostic_questions(
        self,
        vertical: str,
        user_name: str,
    ) -> list[dict]:
        """
        Génère 5 questions de diagnostic structurées en JSON
        """
        vc = VERTICAL_CONFIGS.get(vertical, VERTICAL_CONFIGS["concours"])
        prompt = f"""Tu es un expert du concours de {vc['name']}.
Génère exactement 5 questions de diagnostic pour évaluer les connaissances de {user_name}.
Couvre ces topics : {vc['programme_summary']}

Réponds UNIQUEMENT en JSON valide, sans markdown, avec ce format exact :
{{
  "questions": [
    {{"id": 1, "topic": "nom_du_topic", "question": "texte de la question", "type": "open"}},
    ...
  ]
}}"""

        response = await self._client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )

        content = response.choices[0].message.content
        try:
            data = json.loads(content)
            return data["questions"]
        except (json.JSONDecodeError, KeyError):
            logger.error(f"Erreur parsing JSON diagnostic : {content[:200]}")
            # Fallback sur des questions statiques
            return self._fallback_diagnostic_questions(vertical)

    async def evaluate_diagnostic_answers(
        self,
        questions: list[dict],
        answers: list[dict],
    ) -> dict:
        """
        Évalue les réponses au diagnostic et retourne les scores par topic
        """
        qa_pairs = []
        for q in questions:
            answer = next((a["answer"] for a in answers if a["question_id"] == q["id"]), "")
            qa_pairs.append(f"Q({q['topic']}): {q['question']}\nRéponse: {answer}")

        prompt = f"""Évalue ces réponses à un diagnostic de préparation au concours.
Pour chaque topic, donne un score de 0 à 100.

{chr(10).join(qa_pairs)}

Réponds UNIQUEMENT en JSON valide :
{{
  "scores": {{"topic_name": score_number, ...}},
  "summary": "résumé des lacunes en 2 phrases"
}}"""

        response = await self._client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1,
        )

        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"scores": {}, "summary": "Évaluation non disponible"}

    async def compress_session(self, messages: list[dict]) -> str:
        """
        Compression de contexte : résume la session quand > 2000 tokens
        """
        conversation = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt = f"""Résume cette session de révision en maximum 200 tokens.
Inclure : topics abordés, progrès observés, points à retravailler.

{conversation}"""

        response = await self._client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        return response.choices[0].message.content

    def _fallback_diagnostic_questions(self, vertical: str) -> list[dict]:
        return [
            {"id": 1, "topic": "droit_penal", "question": "Quelle est la durée initiale d'une garde à vue ?", "type": "open"},
            {"id": 2, "topic": "institutions_police", "question": "Sous quelle autorité est placée la DGPN ?", "type": "open"},
            {"id": 3, "topic": "hierarchie_grades", "question": "Citez les grades du corps de maîtrise et d'application dans l'ordre.", "type": "open"},
            {"id": 4, "topic": "droits_citoyens", "question": "Quels sont les droits d'une personne gardée à vue ?", "type": "open"},
            {"id": 5, "topic": "procedure_penale", "question": "À quelles heures peut-on effectuer une perquisition ?", "type": "open"},
        ]


# Singleton de l'agent
maia_agent = MaiaAgent()
