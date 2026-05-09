"""
RAG Service — Retrieval-Augmented Generation
- Embedding : all-MiniLM-L6-v2 (sentence-transformers, local, 384 dims)
- Stockage : pgvector (PostgreSQL)
- Recherche : cosine similarity avec HNSW
"""
import hashlib
import logging
from typing import Optional
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import SentenceTransformer
from pgvector.sqlalchemy import Vector
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
        self._model: Optional[SentenceTransformer] = None

    def _get_model(self) -> SentenceTransformer:
        """Lazy loading du modèle d'embedding"""
        if self._model is None:
            logger.info(f"Chargement du modèle d'embedding : {settings.embedding_model}")
            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Génère un embedding vectoriel pour un texte (384 dims)"""
        model = self._get_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

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

        # Requête pgvector : <=> = distance cosine (1 - similarity)
        # On filtre par topic si précisé (optimisation pre-ANN)
        base_query = """
            SELECT 
                chunk_id,
                topic,
                content,
                1 - (embedding <=> :embedding::vector) AS similarity
            qqqFROM knowledge_chunks
            {topic_filter}
            ORDER BY embedding <=> :embedding::vector
            LIMIT :top_k
        """
        topic_clause = "WHERE topic = :topic" if topic_filter else ""
        sql = text(base_query.format(topic_filter=topic_clause))

        params = {
            "embedding": f"[{','.join(map(str, query_embedding))}]",
            "top_k": top_k,
        }
        if topic_filter:
            params["topic"] = topic_filter

        result = await self._db.execute(sql, params)
        rows = result.fetchall()

        chunks = []
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

    def format_for_prompt(self, chunks: list[dict]) -> str:
        """Formate les chunks pour injection dans le system prompt"""
        if not chunks:
            return ""
        lines = ["Extraits du programme officiel pertinents pour cette question :"]
        for c in chunks:
            lines.append(f"\n[{c['chunk_id']}] {c['content']}")
        return "\n".join(lines)
