"""Endpoints de chat para interação conversacional via streaming SSE.

Este módulo fornece rotas FastAPI para chat em tempo real usando Server-Sent
Events (SSE). As respostas são transmitidas incrementalmente permitindo uma
experiência de chat fluida e responsiva.
"""

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
    """Processa mensagens de chat e retorna resposta via streaming SSE.
    
    Recebe o histórico completo de mensagens, extrai a última mensagem do
    usuário e transmite a resposta do modelo em chunks via Server-Sent Events.
    O histórico anterior (exceto mensagens de sistema) é mantido para contexto.
    
    O formato de resposta segue o padrão OpenAI com estrutura de choices e
    delta para compatibilidade com clientes padrão.

    Args:
        request: Objeto contendo o histórico de mensagens do chat.
        service: Instância do serviço Databricks injetada via dependency
            injection para processar e gerar respostas.

    Returns:
        StreamingResponse configurado com:
            - media_type: "text/event-stream" para SSE
            - headers: Cache-Control e Connection para manter conexão ativa
            - body: Stream de eventos no formato JSON com structure
              `{"choices": [{"delta": {"content": "..."}}]}`

    Raises:
        HTTPException: Status 500 se houver erro durante extração de mensagens,
            processamento do stream ou comunicação com o serviço.

    Note:
        Mensagens com role "system" são filtradas do histórico enviado ao
        modelo. A última mensagem do usuário é separada do histórico para
        processamento otimizado.

    Example:
        Request body:
        ```json
        {
            "messages": [
                {"role": "user", "content": "Olá"},
                {"role": "assistant", "content": "Oi! Como posso ajudar?"},
                {"role": "user", "content": "Explique diabetes"}
            ]
        }
        ```
        
        Response stream (SSE):
        ```
        data: {"choices": [{"delta": {"content": "Diabetes"}}]}
        
        data: {"choices": [{"delta": {"content": " é"}}]}
        
        data: {"choices": [{"delta": {"content": " uma"}}]}
        ```
    """
    try:
        # Extrai a última mensagem do usuário e constrói o histórico a ser enviado
        ultima_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        
        historico = [
            m for m in request.messages 
            if m.content != ultima_msg and m.role != "system"
        ]

        import json

        async def generate():
            """Gera eventos SSE a partir dos chunks do serviço.
            
            Yields:
                Strings formatadas como eventos SSE contendo chunks JSON
                da resposta do modelo.
            """
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