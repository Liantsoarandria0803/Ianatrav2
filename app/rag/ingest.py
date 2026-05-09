"""
Script d'ingestion du corpus RAG
Usage : python -m app.rag.ingest
"""
import asyncio
import logging
from app.core.init_db import ingest_initial_corpus, init_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Démarrage de l'ingestion du corpus RAG...")
    await init_database()
    count = await ingest_initial_corpus()
    logger.info(f"✅ Ingestion terminée : {count} chunks indexés dans pgvector")


if __name__ == "__main__":
    asyncio.run(main())
