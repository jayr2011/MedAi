import httpx
import json
import logging
from typing import AsyncGenerator, List
from app.api.v1.schemas.chat import ChatMessage
from app.core.config import settings
from app.services.rag_service import buscar_contexto
from app.services.web_search_service import web_search, deve_pesquisar_web

logger = logging.getLogger(__name__)

class DatabricksService:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {settings.databricks_token}",
                "Content-Type": "application/json"
            },
            timeout=300.0,
            verify=not settings.debug
        )
        self.endpoint_url = settings.databricks_url

    async def chat_stream(self, question: str, history: List[ChatMessage]) -> AsyncGenerator[str, None]:
        contexto_rag = ""
        contexto_web = ""

        try:
            contexto_rag = buscar_contexto(question)
        except Exception as e:
            logger.error(f"Erro ao buscar contexto RAG: {e}")

        try:
            if deve_pesquisar_web(question):
                logger.info(f"🧠 Roteador decidiu buscar na web para: {question}")
                contexto_web = web_search(question)
        except Exception as e:
            logger.error(f"Erro na busca web: {e}")

        system_prompt = (
            "Você é o MedAi, um assistente médico inteligente e empático. "
            "Sua tarefa é fornecer informações baseadas em evidências.\n\n"
        )

        if contexto_rag:
            system_prompt += f"--- CONTEXTO DOS SEUS DOCUMENTOS ---\n{contexto_rag}\n\n"

        if contexto_web:
            system_prompt += f"--- CONTEXTO ATUALIZADO DA WEB ---\n{contexto_web}\n\n"

        system_prompt += (
            "Importante: Se as informações acima forem conflitantes, priorize os documentos locais. "
            "Sempre cite a fonte e o número da página (ex: Fonte X, pág. Y) imediatamente após a informação extraída dos documentos locais."
        )

        messages_payload = [{"role": "system", "content": system_prompt}]

        for msg in history:
            messages_payload.append({"role": msg.role, "content": msg.content})
        
        messages_payload.append({"role": "user", "content": question})

        payload = {
            "messages": messages_payload,
            "max_tokens": settings.max_tokens or 1024,
            "temperature": 0.2,
            "stream": True,
        }

        async with self.client.stream("POST", self.endpoint_url, json=payload) as response:
            if response.status_code != 200:
                error = await response.aread()
                raise ValueError(f"Databricks {response.status_code}: {error.decode()}")

            async for line in response.aiter_lines():
                stripped = line.strip()
                if stripped.startswith("data: "):
                    data = stripped[6:]
                    if data and data != "[DONE]":
                        try:
                            json_data = json.loads(data)
                            content = json_data['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError):
                            continue