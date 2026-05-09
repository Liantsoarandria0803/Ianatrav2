from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.schemas.schemas import DiagnosticStartResponse, DiagnosticSubmitRequest, DiagnosticSubmitResponse
from app.services.services import DiagnosticService

router = APIRouter(prefix="/diagnostic", tags=["diagnostic"])


@router.post("/start", response_model=DiagnosticStartResponse)
async def start_diagnostic(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    try:
        service = DiagnosticService(db)
        result = await service.start(UUID(user_id))
        return DiagnosticStartResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/submit", response_model=DiagnosticSubmitResponse)
async def submit_diagnostic(
    req: DiagnosticSubmitRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    try:
        service = DiagnosticService(db)
        result = await service.submit(UUID(user_id), req.diagnostic_id, [a.model_dump() for a in req.answers])
        return DiagnosticSubmitResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
