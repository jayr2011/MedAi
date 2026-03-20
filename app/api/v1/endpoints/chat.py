import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.api.v1.schemas.chat import ChatRequest
from app.services.llm_service import LlmService
from app.api.deps import get_llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: LlmService = Depends(get_llm_service)
):
    try:
        ultima_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        
        historico = request.history or []

        import json

        async def generate():
            try:
                async for chunk in service.chat_stream(ultima_msg, historico):
                    payload = json.dumps({"choices": [{"delta": {"content": chunk}}]})
                    yield f"data: {payload}\n\n"
            except Exception:
                logger.exception("Erro durante o streaming SSE")
                error_payload = json.dumps(
                    {"error": {"message": "Falha durante a geração da resposta."}}
                )
                yield f"data: {error_payload}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    except Exception as e:
        logger.error(f"Erro na rota de chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
