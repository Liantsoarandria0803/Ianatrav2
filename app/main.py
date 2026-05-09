import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.api.v1.endpoints import auth, diagnostic, profile, session
from app.core.config import get_settings
from app.core.database import engine
from app.core.init_db import ingest_initial_corpus, init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


async def bootstrap_database() -> None:
    """Initialise la DB sans bloquer l'ouverture du port Render."""
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            await init_database()
            await ingest_initial_corpus()
            logger.info("✅ Base de données et corpus RAG prêts")
            return
        except Exception:
            logger.exception(
                "Initialisation DB/RAG échouée (tentative %s/%s)",
                attempt,
                max_attempts,
            )
            if attempt < max_attempts:
                await asyncio.sleep(min(5 * attempt, 30))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup : ouvrir l'API rapidement, puis préparer la DB en arrière-plan."""
    logger.info("🚀 MAÏA backend démarrage...")
    bootstrap_task = asyncio.create_task(bootstrap_database())
    try:
        yield
    finally:
        if not bootstrap_task.done():
            bootstrap_task.cancel()
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

# Compatibility routes for frontends configured with the backend root URL
# instead of the versioned API base URL.
app.include_router(auth.router)
app.include_router(diagnostic.router)
app.include_router(session.router)
app.include_router(profile.router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "maia-backend", "api": "/api/v1"}


@app.head("/")
async def root_head():
    return None


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "maia-backend", "version": "1.0.0"}
