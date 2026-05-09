"""
Script d'ingestion du corpus RAG
Usage : python -m app.rag.ingest
"""
import asyncio
import logging
from app.core.database import AsyncSessionLocal
from app.core.init_db import init_database
from app.rag.rag_service import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Démarrage de l'ingestion du corpus RAG...")
    await init_database()
    async with AsyncSessionLocal() as db:
        service = RAGService(db)
        count = await service.ingest()
        await db.commit()
        logger.info(f"✅ Ingestion terminée : {count} chunks indexés dans pgvector")


if __name__ == "__main__":
    asyncio.run(main())
