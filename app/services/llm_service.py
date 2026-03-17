import logging
from typing import AsyncGenerator, List

from llama_cpp import Llama

from app.api.v1.schemas.chat import ChatMessage
from app.core.config import settings
from app.services.rag_service import buscar_contexto
from app.services.web_search_service import web_search, deve_pesquisar_web

logger = logging.getLogger(__name__)


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
            logger.info("Modelo de chat BioMistral carregado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de chat BioMistral: {e}")
            self.chat_llm = None

        self.guardrail_llm = None
        try:
            self.guardrail_llm = Llama.from_pretrained(
                repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                filename="Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf",
                n_ctx=1024,
                n_threads=4,
                verbose=False
            )
            logger.info("Guardrail Llama-3 carregado com sucesso na CPU.")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo Guardrail Llama-3: {e}")
            self.guardrail_llm = None

    async def is_pergunta_medica(self, question: str) -> bool:
        if not self.guardrail_llm:
            logger.warning("Guardrail indisponível, assumindo que a pergunta é médica.")
            return True

        prompt = (
            f"Você é um classificador que responde apenas SIM ou NÃO.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"A seguinte pergunta ou descrição é sobre saúde, medicina ou biologia humana?\n\n"
            f"'{question}'\n\n"
            f"Responda apenas: SIM ou NÃO<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        try:
            output = self.guardrail_llm(
                prompt,
                max_tokens=10,
                stop=["<|eot_id|>"],
                temperature=0.2
            )
            resposta = output["choices"][0]["text"].strip().upper()
            return "SIM" in resposta or "YES" in resposta
        except Exception as e:
            logger.error(f"Erro ao classificar pergunta: {e}")
            return True

    async def chat_stream(
        self,
        question: str,
        history: List[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        if not await self.is_pergunta_medica(question):
            yield "Peço desculpa, mas como MedAi, só posso responder a questões relacionadas com saúde e medicina."
            return

        contexto_rag = ""
        contexto_web = ""

        try:
            contexto_rag = buscar_contexto(question)
        except Exception as e:
            logger.error(f"Erro ao buscar contexto RAG: {e}")

        try:
            if deve_pesquisar_web(question):
                logger.info(f"Roteador decidiu buscar na web para: {question}")
                contexto_web = web_search(question)
        except Exception as e:
            logger.error(f"Erro na busca web: {e}")

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

            # Streaming local via llama_cpp com interface de chat compatível.
            stream = self.chat_llm.create_chat_completion(
                messages=messages,
                max_tokens=settings.max_tokens or 1024,
                temperature=0.2,
                stream=True,
            )

            for chunk in stream:
                choices = chunk.get("choices", []) if isinstance(chunk, dict) else []
                if not choices:
                    continue

                delta = choices[0].get("delta", {}) or {}
                content = delta.get("content")

                if not content:
                    content = choices[0].get("text")

                if content:
                    yield content

        except Exception as e:
            logger.exception("Erro ao consultar provedor LLM")
            raise ValueError(f"Erro ao consultar provedor LLM: {e}") from e
