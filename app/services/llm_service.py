import logging
import asyncio
import time
from typing import AsyncGenerator, List

from llama_cpp import Llama

from app.api.v1.schemas.chat import ChatMessage
from app.core.config import settings
from app.services.rag_service import buscar_contexto
from app.services.web_search_service import web_search, deve_pesquisar_web

logger = logging.getLogger(__name__)

MEDICAL_HINTS = (
    "dor", "febre", "sintoma", "diagnost", "tratament", "exame", "medic",
    "pressao", "glic", "insulina", "diabetes", "cancer", "oncolog", "cefale",
    "dispne", "asma", "covid", "infec", "cardio", "renal", "hepatic", "neuro",
)


class LlmService:
    def __init__(self) -> None:
        self.chat_llm = None
        try:
            self.chat_llm = Llama.from_pretrained(
                repo_id=settings.llm_repo_id,
                filename=settings.llm_filename,
                n_ctx=settings.max_tokens or 4096,
                n_threads=settings.llm_threads,
                verbose=False,
            )
            logger.info("Modelo local carregado com sucesso: %s/%s", settings.llm_repo_id, settings.llm_filename)
        except Exception as e:
            logger.error("Erro ao carregar modelo local: %s", e)
            self.chat_llm = None

    def _eh_obviamente_medica(self, question: str) -> bool:
        q = question.lower()
        return any(hint in q for hint in MEDICAL_HINTS)

    async def _coletar_contextos(self, question: str) -> tuple[str, str]:
        async def rag_task() -> str:
            try:
                return await asyncio.to_thread(buscar_contexto, question)
            except Exception as e:
                logger.error("Erro ao buscar contexto RAG: %s", e)
                return ""

        async def web_task() -> str:
            try:
                should_search = await asyncio.to_thread(deve_pesquisar_web, question)
                if should_search:
                    logger.info("Roteador decidiu buscar na web para: %s", question)
                    return await asyncio.to_thread(web_search, question, 5)
            except Exception as e:
                logger.error("Erro na busca web: %s", e)
            return ""

        return await asyncio.gather(rag_task(), web_task())

    async def is_pergunta_medica(self, question: str) -> bool:
        if self._eh_obviamente_medica(question):
            return True

        if not self.chat_llm:
            logger.warning("Modelo local indisponível, assumindo que a pergunta é médica.")
            return True

        try:
            messages = [
                {
                    "role": "system",
                    "content": "Você é um classificador que responde apenas SIM ou NÃO.",
                },
                {
                    "role": "user",
                    "content": (
                        "A seguinte pergunta ou descrição é sobre saúde, medicina "
                        f"ou biologia humana?\n\n'{question}'\n\nResponda apenas: SIM ou NÃO"
                    ),
                },
            ]
            output = self.chat_llm.create_chat_completion(
                messages=messages,
                max_tokens=8,
                temperature=0.0,
            )
            resposta = output["choices"][0].get("message", {}).get("content", "").strip().upper()
            return "SIM" in resposta or "YES" in resposta
        except Exception as e:
            logger.error(f"Erro ao classificar pergunta: {e}")
            return True

    async def chat_stream(
        self,
        question: str,
        history: List[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        started_at = time.perf_counter()

        if not await self.is_pergunta_medica(question):
            yield "Peço desculpa, mas como MedAi, só posso responder a questões relacionadas com saúde e medicina."
            return

        contexto_rag, contexto_web = await self._coletar_contextos(question)

        system_prompt = (
            "Você é o MedAi, um assistente médico inteligente. "
            "Você fala com médicos registrados no CRM, não pacientes. "
            "Sua tarefa é fornecer informações baseadas em evidências.\n\n"
            "Sempre me dê 5 possíveis diagnósticos ou tratamentos relacionados à pergunta, "
            "mesmo que sejam apenas possibilidades remotas, e exames complementares para investigação, "
            "explicando por que cada um deles é relevante. "
            "Se possível, inclua referências bibliográficas confiáveis.\n\n"
            "Sempre cite a fonte quando usar informações dos documentos locais e, se possível, "
            "inclua o número da página."
        )

        if contexto_rag:
            system_prompt += f"\n\n--- CONTEXTO DOS SEUS DOCUMENTOS ---\n{contexto_rag}\n"

        if contexto_web:
            system_prompt += f"\n\n--- CONTEXTO ATUALIZADO DA WEB ---\n{contexto_web}\n"

        system_prompt += (
            "\nImportante: se houver conflito, priorize os documentos locais. "
            "Sempre cite a fonte e a página imediatamente após a informação extraída."
        )

        messages = [{"role": "system", "content": system_prompt}]

        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": question})

        try:
            if not self.chat_llm:
                raise ValueError("Modelo local de chat indisponível.")

            stream = self.chat_llm.create_chat_completion(
                messages=messages,
                max_tokens=settings.max_tokens or 1024,
                temperature=0.2,
                stream=True,
            )

            first_token_logged = False
            for chunk in stream:
                choices = chunk.get("choices", []) if isinstance(chunk, dict) else []
                if not choices:
                    continue

                delta = choices[0].get("delta", {}) or {}
                content = delta.get("content")
                if content:
                    if not first_token_logged:
                        logger.info("Tempo até 1o token: %.2fs", time.perf_counter() - started_at)
                        first_token_logged = True
                    yield content

        except Exception as e:
            logger.exception("Erro ao consultar provedor LLM")
            raise ValueError(f"Erro ao consultar provedor LLM: {e}") from e
