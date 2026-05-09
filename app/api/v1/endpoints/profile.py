from uuid import UUID
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.schemas.schemas import CompetenceProfileResponse
from app.services.services import ProfileService

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("/competences", response_model=CompetenceProfileResponse)
async def get_competences(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    service = ProfileService(db)
    result = await service.get_competences(UUID(user_id))
    return CompetenceProfileResponse(**result)
