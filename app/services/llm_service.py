import logging
import asyncio
from typing import AsyncGenerator, List
from google import genai
from google.genai import types

from app.api.v1.schemas.chat import ChatMessage
from app.core.config import settings
from app.services.rag_service import buscar_contexto

logger = logging.getLogger(__name__)
MEDICAL_HINTS = (
    "dor", "febre", "sintoma", "diagnost", "tratament", "exame", "medic",
    "pressao", "glic", "insulina", "diabetes", "cancer", "oncolog", "cefale",
    "dispne", "asma", "covid", "infec", "cardio", "renal", "hepatic", "neuro",
)

class LlmService:
    def __init__(self) -> None:
        try:
            self.client = genai.Client(api_key=settings.gemini_api_key)
            self.model_name = settings.llm_model
            logger.info("Gemini Client inicializado com sucesso.")
        except Exception as e:
            logger.error("Erro ao inicializar Gemini Client: %s", e)
            self.client = None
            self.model_name = "gemini-2.5-flash-lite"

    async def _coletar_contextos(self, question: str) -> tuple[str, str]:
        """Coleta contexto local do RAG.

        Busca web fica desativada aqui para evitar dependências quebradas
        durante a migração para Gemini.
        """
        try:
            contexto_rag = await asyncio.to_thread(buscar_contexto, question)
        except Exception as e:
            logger.error("Erro ao buscar contexto RAG: %s", e)
            contexto_rag = ""
        return contexto_rag, ""
    
    async def chat_stream(
        self,
        question: str,
        history: List[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        if self.client is None:
            raise ValueError("Cliente Gemini não inicializado.")
        
        contexto_rag, contexto_web = await self._coletar_contextos(question)
        
        system_instruction = (
            "Você é o MedAi, um assistente médico inteligente. "
            "Você fala com médicos registrados no CRM, não pacientes. "
            "Sua tarefa é fornecer informações baseadas em evidências.\n\n"
            "Sempre me dê 5 possíveis diagnósticos ou tratamentos relacionados à pergunta, "
            "mesmo que sejam apenas possibilidades remotas, e exames complementares para investigação, "
            "explicando por que cada um deles é relevante. "
            "Se possível, inclua referências bibliográficas confiáveis.\n\n"
        )
        
        if contexto_rag:
            system_instruction += f"\n\n--- CONTEXTO DOS SEUS DOCUMENTOS ---\n{contexto_rag}\n"
        if contexto_web:
            system_instruction += f"\n\n--- CONTEXTO ATUALIZADO DA WEB ---\n{contexto_web}\n"

        system_instruction += "\nImportante: se houver conflito, priorize os documentos locais. Sempre cite a fonte e a página imediatamente após a informação extraída."
        
        use_developer_instruction = "gemma" not in self.model_name.lower()
        if use_developer_instruction:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=settings.max_tokens or 1024,
            )
        else:
            config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=settings.max_tokens or 1024,
            )
        
        formatted_history = []
        
        for msg in history:
            role = "user" if msg.role == "user" else "model"
            formatted_history.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg.content)])
            )

        if not use_developer_instruction:
            formatted_history.insert(
                0,
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=f"Instruções para resposta:\n{system_instruction}"
                        )
                    ],
                ),
            )
            
        formatted_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=question)])
        )
        
        try:
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=formatted_history,
                config=config
            )

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.exception("Erro ao consultar provedor LLM")
            raise ValueError(f"Erro ao consultar provedor LLM: {e}") from e
