from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.api.v1.schemas.chat import ChatRequest, ChatMessage
from app.services.databricks_service import DatabricksService
from app.services.rag_service import buscar_contexto
from app.api.deps import get_databricks_service

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: DatabricksService = Depends(get_databricks_service)
):
    try:
        # Última mensagem do usuário
        ultima_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )

        # Buscar contexto RAG
        contexto = buscar_contexto(ultima_msg)

        # Injetar contexto como system message
        messages = list(request.messages)
        if contexto:
            msg_rag = ChatMessage(
                role="system",
                content=(
                    "Use o seguinte contexto dos livros médicos para responder. "
                    "Cite a fonte quando possível. "
                    "Se a informação não estiver no contexto, responda com seu conhecimento geral.\n\n"
                    f"CONTEXTO:\n{contexto}"
                )
            )
            # Inserir após o system message original
            messages.insert(1, msg_rag)

        async def generate():
            async for chunk in service.chat_stream(messages):
                yield f"data: {chunk}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Databricks error: {str(e)}")