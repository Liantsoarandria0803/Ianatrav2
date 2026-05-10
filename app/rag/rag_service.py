"""
RAG Service — Retrieval-Augmented Generation

Objectif: fournir du contexte au prompt sans rendre l'API fragile en prod.

Stratégie:
- Mode "vector" (si embeddings dispo) : cosine similarity via pgvector.
- Fallback "keyword" (toujours dispo) : full-text search Postgres.

Note: Sur des environnements à ressources limitées (ex: Render free),
les embeddings locaux (torch) peuvent être trop lents/lourds. Le fallback
permet de garder le chat fonctionnel.
"""
import logging
import hashlib
import math
import re
from typing import Any, Optional
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import KnowledgeChunk, VerticalEnum
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Corpus de test MAÏA (défini dans le cahier des charges)
INITIAL_CORPUS = [
    {
        "chunk_id": "C001",
        "topic": "droit_penal",
        "content": "Le gardien de la paix peut procéder à une garde à vue d'une durée initiale de 24h, renouvelable une fois sur autorisation du procureur. La garde à vue est notifiée immédiatement à la personne concernée.",
    },
    {
        "chunk_id": "C002",
        "topic": "institutions_police",
        "content": "La Direction Générale de la Police Nationale (DGPN) est placée sous l'autorité du ministre de l'Intérieur. Elle comprend la Direction Centrale de la Sécurité Publique (DCSP) et la Direction Centrale de la Police Judiciaire (DCPJ).",
    },
    {
        "chunk_id": "C003",
        "topic": "hierarchie_grades",
        "content": "La hiérarchie du corps de maîtrise et d'application est : Gardien stagiaire → Gardien de la paix → Brigadier → Brigadier-chef → Major. L'accès au grade de Brigadier se fait par concours interne ou examen professionnel.",
    },
    {
        "chunk_id": "C004",
        "topic": "droits_citoyens",
        "content": "Tout citoyen interpellé a le droit d'être informé des faits qui lui sont reprochés, de garder le silence, d'être assisté d'un avocat dès la première heure de garde à vue, et de faire prévenir un proche.",
    },
    {
        "chunk_id": "C005",
        "topic": "procedure_penale",
        "content": "La perquisition ne peut être effectuée qu'entre 6h et 21h sauf flagrant délit ou enquête préliminaire. Elle nécessite l'accord de l'occupant ou une autorisation du juge d'instruction.",
    },
]


class RAGService:
    """
    Service RAG : embedding + stockage pgvector + recherche sémantique
    SOLID S : responsabilité unique = gestion du corpus vectoriel
    """

    def __init__(self, db: AsyncSession):
        self._db = db
        self._model: Any = None
        self._embeddings_available: Optional[bool] = None

        # 384 dims pour coller au schéma pgvector Vector(384)
        self._fallback_dim = 384

    def _load_embedding_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            logger.info("Embeddings désactivés: sentence-transformers non installé")
            return None

        try:
            logger.info("Chargement du modèle d'embedding : %s", settings.embedding_model)
            return SentenceTransformer(settings.embedding_model)
        except Exception:
            logger.warning("Échec chargement embeddings; fallback keyword activé", exc_info=True)
            return None

    def _hash_embed(self, text: str) -> list[float]:
        """Embedding léger (sans dépendances) basé sur feature hashing.

        Objectif: fournir un vecteur stable 384-dim pour pgvector même quand les
        embeddings sémantiques ne sont pas disponibles.
        """
        tokens = [t for t in re.split(r"\W+", (text or "").lower()) if t]
        vec = [0.0] * self._fallback_dim

        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self._fallback_dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _get_model(self) -> Any:
        """Lazy loading du modèle d'embedding (optionnel)."""
        if self._embeddings_available is False:
            return None

        if self._model is None:
            self._model = self._load_embedding_model()
            self._embeddings_available = self._model is not None

        return self._model

    def embed(self, text: str) -> list[float]:
        """Génère un embedding 384-dim.

        - Si sentence-transformers est dispo: embedding sémantique.
        - Sinon: fallback hashing (lexical).
        """
        model = self._get_model()
        if model is None:
            return self._hash_embed(text)

        try:
            vector = model.encode(text, normalize_embeddings=True)
            if hasattr(vector, "tolist"):
                return vector.tolist()
            return list(vector)
        except Exception:
            # Désactiver pour le reste du process: évite de retenter à chaque requête.
            logger.warning("Échec génération embedding; fallback keyword activé", exc_info=True)
            self._model = None
            self._embeddings_available = False
            return self._hash_embed(text)

    async def ingest(self, chunks: list[dict] = None) -> int:
        """
        Pipeline d'ingestion : texte → embedding → pgvector
        Retourne le nombre de chunks ingérés
        """
        if chunks is None:
            chunks = INITIAL_CORPUS

        count = 0
        for chunk_data in chunks:
            # Vérifier si le chunk existe déjà
            result = await self._db.execute(
                select(KnowledgeChunk).where(KnowledgeChunk.chunk_id == chunk_data["chunk_id"])
            )
            existing = result.scalar_one_or_none()
            if existing:
                logger.info(f"Chunk {chunk_data['chunk_id']} déjà indexé, skip")
                continue

            embedding = self.embed(chunk_data["content"])
            chunk = KnowledgeChunk(
                chunk_id=chunk_data["chunk_id"],
                topic=chunk_data["topic"],
                content=chunk_data["content"],
                embedding=embedding,
                vertical=VerticalEnum.concours,
            )
            self._db.add(chunk)
            count += 1
            logger.info(f"Chunk {chunk_data['chunk_id']} indexé (topic: {chunk_data['topic']})")

        await self._db.flush()
        logger.info(f"Ingestion terminée : {count} nouveaux chunks")
        return count

    async def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.7,
        topic_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Recherche sémantique par cosine similarity avec pgvector
        Retourne les chunks les plus similaires avec leur score
        """
        query_embedding = self.embed(query)

        effective_min_similarity = min_similarity
        # Le fallback hashing donne des similarités plus basses que les modèles sémantiques.
        if self._embeddings_available is False:
            effective_min_similarity = min(min_similarity, 0.15)
        try:
            return await self._vector_search(
                query_embedding,
                top_k=top_k,
                min_similarity=effective_min_similarity,
                topic_filter=topic_filter,
            )
        except Exception:
            logger.warning("Vector search échoué; fallback keyword activé", exc_info=True)
            return await self._keyword_search(query, top_k=top_k, topic_filter=topic_filter)

    async def _vector_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        min_similarity: float,
        topic_filter: Optional[str],
    ) -> list[dict]:
        """Recherche vectorielle pgvector (cosine)."""

        # Requête pgvector : <=> = distance cosine (1 - similarity)
        # On filtre par topic si précisé (optimisation pre-ANN)
        base_query = """
            SELECT 
                chunk_id,
                topic,
                content,
                1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM knowledge_chunks
            WHERE embedding IS NOT NULL
            {topic_filter}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """
        topic_clause = "AND topic = :topic" if topic_filter else ""
        sql = text(base_query.format(topic_filter=topic_clause))

        params = {
            "embedding": f"[{','.join(map(str, query_embedding))}]",
            "top_k": top_k,
        }
        if topic_filter:
            params["topic"] = topic_filter

        result = await self._db.execute(sql, params)
        rows = result.fetchall()

        chunks: list[dict] = []
        for row in rows:
            if row.similarity >= min_similarity:
                chunks.append({
                    "chunk_id": row.chunk_id,
                    "topic": row.topic,
                    "content": row.content,
                    "similarity": round(float(row.similarity), 4),
                })
            else:
                logger.info(
                    f"Chunk {row.chunk_id} ignoré (similarity={row.similarity:.3f} < {min_similarity})"
                )

        logger.info(f"RAG search '{query[:50]}...' → {len(chunks)}/{top_k} chunks retenus")
        return chunks

    async def _keyword_search(
        self,
        query: str,
        *,
        top_k: int,
        topic_filter: Optional[str],
    ) -> list[dict]:
        """Fallback robuste sans embeddings: full-text search Postgres."""
        q = (query or "").strip()
        if not q:
            return []

        base_query = """
            SELECT
                chunk_id,
                topic,
                content,
                ts_rank_cd(
                    to_tsvector('simple', content),
                    plainto_tsquery('simple', :q)
                ) AS similarity
            FROM knowledge_chunks
            WHERE to_tsvector('simple', content) @@ plainto_tsquery('simple', :q)
            {topic_filter}
            ORDER BY similarity DESC
            LIMIT :top_k
        """
        topic_clause = "AND topic = :topic" if topic_filter else ""
        sql = text(base_query.format(topic_filter=topic_clause))
        params = {"q": q, "top_k": top_k}
        if topic_filter:
            params["topic"] = topic_filter

        result = await self._db.execute(sql, params)
        rows = result.fetchall()

        chunks: list[dict] = []
        for row in rows:
            chunks.append({
                "chunk_id": row.chunk_id,
                "topic": row.topic,
                "content": row.content,
                "similarity": round(float(row.similarity or 0.0), 4),
            })

        logger.info("RAG keyword search '%s...' → %s/%s chunks", q[:50], len(chunks), top_k)
        return chunks

    def format_for_prompt(self, chunks: list[dict]) -> str:
        """Formate les chunks pour injection dans le system prompt"""
        if not chunks:
            return ""
        lines = ["Extraits du programme officiel pertinents pour cette question :"]
        for c in chunks:
            lines.append(f"\n[{c['chunk_id']}] {c['content']}")
        return "\n".join(lines)
