"""
Session endpoints — le streaming SSE est le critère technique le plus important
"""
import json
import logging
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db, AsyncSessionLocal
from app.core.security import get_current_user_id
from app.schemas.schemas import SessionStartRequest, SessionStartResponse, MessageRequest, SessionHistoryResponse, MessageResponse
from app.services.services import SessionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/session", tags=["session"])


@router.post("/start", response_model=SessionStartResponse, status_code=status.HTTP_201_CREATED)
async def start_session(
    req: SessionStartRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    service = SessionService(db)
    session = await service.start_session(UUID(user_id), req.mode, req.topic)
    # Mapping explicite ORM -> API (évite les soucis d'alias Pydantic avec from_attributes)
    return SessionStartResponse(
        session_id=session.id,
        mode=session.mode,
        topic=session.topic,
        started_at=session.started_at,
    )


@router.post("/{session_id}/message")
async def send_message(
    session_id: UUID,
    req: MessageRequest,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """
    Streaming SSE — les tokens apparaissent en temps réel côté client
    Chaque chunk est au format : data: <token>\n\n
    Le flux se termine par : data: [DONE]\n\n
    """

    async def event_stream():
        # Utiliser une nouvelle session DB pour le streaming (la session FastAPI est fermée)
        async with AsyncSessionLocal() as db:
            try:
                service = SessionService(db)
                token_count = 0

                async for token in service.stream_message(session_id, UUID(user_id), req.content):
                    # Format SSE standard
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    token_count += 1

                await db.commit()
                # Signal de fin de stream
                yield f"data: {json.dumps({'done': True, 'tokens': token_count})}\n\n"

            except ValueError as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except Exception as e:
                logger.error(f"Erreur streaming session {session_id} : {e}", exc_info=True)
                await db.rollback()
                yield f"data: {json.dumps({'error': 'Erreur interne du serveur'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Important pour Nginx/Render
            "Connection": "keep-alive",
        },
    )


@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    try:
        service = SessionService(db)
        result = await service.get_history(session_id, UUID(user_id))
        return SessionHistoryResponse(
            session_id=result["session_id"],
            messages=[MessageResponse.model_validate(m) for m in result["messages"]],
            total_tokens=result["total_tokens"],
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
