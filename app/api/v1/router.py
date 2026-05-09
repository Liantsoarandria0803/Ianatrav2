from fastapi import APIRouter
from app.api.v1.endpoints import auth, diagnostic, session, profile

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth.router)
api_router.include_router(diagnostic.router)
api_router.include_router(session.router)
api_router.include_router(profile.router)
