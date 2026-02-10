from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.api.v1.schemas.chat import ChatRequest, ChatMessage
from app.services.databricks_service import DatabricksService
from app.services.rag_service import buscar_contexto
from app.api.deps import get_databricks_service
from app.services.web_search_service import web_search, deve_pesquisar_web

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: DatabricksService = Depends(get_databricks_service)
):
    try:
        ultima_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )

        contexto_rag = buscar_contexto(ultima_msg)

        contexto_web = ""

        precisa_web = deve_pesquisar_web(ultima_msg)

        if precisa_web or (not contexto_rag and len(ultima_msg) > 15):
            contexto_web = web_search(ultima_msg)

        system_instructions = "Você é a MedIA. responda sempre em português brasileiro. sempre indique entre 3 a 5 possiveis causas para os sintomas apresentados. indique exames complementares para confirmar o diagnóstico."

        if contexto_rag:
            system_instructions += f"\n\nCONTEXTO DOS LIVROS:\n{contexto_rag}\n"

        if contexto_web:
            system_instructions += f"\n\nRESULTADOS DA WEB (Complementar):\n{contexto_web}\nCite a fonte (link) se usar a web."

        if not contexto_rag and not contexto_web:
            system_instructions += "Responda com seu conhecimento base."
        
        system_instructions += "\n\nSe não souber a resposta, diga que não sabe."

        messages = list(request.messages)
        
        messages = [m for m in messages if m.role != "system"]

        msg_system = ChatMessage(role="system", content=system_instructions)
        messages.insert(0, msg_system)

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