import logging

from sqlalchemy import text

from app.core.database import AsyncSessionLocal, Base, engine

logger = logging.getLogger(__name__)


async def init_database() -> None:
    """Initialise pgvector, les tables SQLAlchemy et l'index vectoriel."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw
            ON knowledge_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
    logger.info("Base de donnees initialisee avec pgvector + index HNSW")


async def ingest_initial_corpus() -> int:
    """Ingère le corpus RAG initial si les chunks ne sont pas déjà présents."""
    from app.rag.rag_service import RAGService

    async with AsyncSessionLocal() as db:
        service = RAGService(db)
        count = await service.ingest()
        await db.commit()
        logger.info("Corpus RAG initialise: %s nouveaux chunks", count)
        return count
