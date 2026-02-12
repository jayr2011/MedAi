import httpx
import json
import logging
from typing import AsyncGenerator, List
from app.api.v1.schemas.chat import ChatMessage
from app.core.config import settings
from app.services.rag_service import buscar_contexto
from app.services.web_search_service import web_search, deve_pesquisar_web
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class DatabricksService:
    """Serviço para interagir com a API do Databricks e realizar operações relacionadas ao modelo de linguagem e classificação de perguntas médicas."""
    def __init__(self) -> None:
        """Inicializa o cliente HTTP para comunicação com o Databricks e tenta carregar um modelo local de Llama para classificação de perguntas médicas, caso esteja disponível."""
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {settings.databricks_token}",
                "Content-Type": "application/json"
            },
            timeout=300.0,
            verify=not settings.debug
        )
        self.endpoint_url = settings.databricks_url
        self.guardrail_llm = None
        try:
            """Carrega o modelo Guardrail Llama-3 localmente para classificação de perguntas médicas, evitando custos de token do Databricks para essa tarefa. O modelo é otimizado para rodar na CPU, garantindo acessibilidade mesmo sem GPU dedicada."""
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
        """Verifica escopo localmente na CPU sem gastar tokens do Databricks"""
        if not self.guardrail_llm:
            logger.warning("Guardrail Llama-3 não disponível, assumindo que a pergunta é médica.")
            return True
        
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"Responda apenas SIM ou NÃO.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"A pergunta '{question}' é sobre saúde, medicina ou biologia humana?<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        try:
            """Executa o modelo Guardrail Llama-3 para classificar a pergunta, esperando uma resposta clara de SIM ou NÃO. A resposta é processada para determinar se a pergunta é médica ou não, e o resultado é logado para monitoramento e análise futura."""
            output = self.guardrail_llm(prompt, max_tokens=5, stop=["<|eot_id|>"], temperature=0.0)
            resposta = output["choices"][0]["text"].strip().upper()

            is_medical = resposta == "SIM"
            logger.info(f"Guardrail: '{question}' -> {resposta} (Médica: {is_medical})")
            return is_medical
        except Exception as e:
            logger.error(f"Erro ao classificar a pergunta com Guardrail Llama-3: {e}")
            return True 

    async def chat_stream(self, question: str, history: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Gera uma resposta em streaming do Databricks, integrando contexto RAG e resultados de busca web quando aplicável."""
        if not await self.is_pergunta_medica(question):
            yield "Peço desculpa, mas como MedAi, só posso responder a questões relacionadas com saúde e medicina. Como posso ajudar com o seu bem-estar hoje?"
            return

        contexto_rag = ""
        contexto_web = ""

        try:
            """Busca contexto relevante usando RAG para a pergunta, o que pode incluir informações de documentos locais ou bases de conhecimento pré-indexadas."""
            contexto_rag = buscar_contexto(question)
        except Exception as e:
            logger.error(f"Erro ao buscar contexto RAG: {e}")

        try:
            """Pode ser necessário realizar uma busca na web para obter informações atualizadas ou complementares, especialmente se o contexto local for insuficiente. A decisão de buscar na web é baseada em uma função que avalia a pergunta e o contexto disponível."""
            if deve_pesquisar_web(question):
                logger.info(f"🧠 Roteador decidiu buscar na web para: {question}")
                contexto_web = web_search(question)
        except Exception as e:
            logger.error(f"Erro na busca web: {e}")

        system_prompt = (
            "Você é o MedAi, um assistente médico inteligente de uma apresentação curta. Você fala com medicos registrados no CRM não pacientes."
            "Sua tarefa é fornecer informações baseadas em evidências.\n\n"
            "sempre me de 5 possíveis diagnósticos ou tratamentos relacionados à pergunta, mesmo que sejam apenas possibilidades remotas e exames complementares para investigação, e explique o porquê de cada um deles ser relevante para a pergunta. Se possível, inclua referências bibliográficas confiáveis para cada diagnóstico ou tratamento sugerido. "
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
            """Processa a resposta em streaming do Databricks, extraindo e yieldando o conteúdo à medida que chega."""
            if response.status_code != 200:
                error = await response.aread()
                raise ValueError(f"Databricks {response.status_code}: {error.decode()}")

            async for line in response.aiter_lines():
                """Cada linha do stream é esperada no formato 'data: {json}', onde o JSON contém o conteúdo gerado."""
                stripped = line.strip()
                if stripped.startswith("data: "):
                    data = stripped[6:]
                    if data and data != "[DONE]":
                        try:
                            """Tenta decodificar o JSON da linha para extrair o conteúdo gerado. O conteúdo é esperado no campo 'choices[0].delta.content'. Se o JSON estiver malformado ou não contiver os campos esperados, a linha é ignorada para evitar interrupções no stream."""
                            json_data = json.loads(data)
                            content = json_data['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError):
                            continue