import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.database import engine
from app.core.init_db import init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup : créer les tables et activer pgvector"""
    logger.info("🚀 MAÏA backend démarrage...")
    await init_database()
    logger.info("✅ Base de données initialisée avec pgvector + index HNSW")
    yield
    logger.info("👋 MAÏA backend arrêt")
    await engine.dispose()


app = FastAPI(
    title="MAÏA — Module de Session Pédagogique IA",
    description="API backend pour la plateforme d'apprentissage adaptatif MAÏA",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "maia-backend", "version": "1.0.0"}
