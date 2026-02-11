import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.api.v1.schemas.chat import ChatRequest
from app.services.databricks_service import DatabricksService
from app.api.deps import get_databricks_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: DatabricksService = Depends(get_databricks_service)
):
    """Endpoint para chat com streaming de resposta usando Server-Sent Events (SSE)"""
    try:
        ultima_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        
        historico = [
            m for m in request.messages 
            if m.content != ultima_msg and m.role != "system"
        ]

        import json

        async def generate():
            async for chunk in service.chat_stream(ultima_msg, historico):
                payload = json.dumps({"choices": [{"delta": {"content": chunk}}]})
                yield f"data: {payload}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    except Exception as e:
        logger.error(f"Erro na rota de chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")