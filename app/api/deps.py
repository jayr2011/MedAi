"""Dependências compartilhadas do aplicativo.

Este módulo expõe factories/dependencies usadas pelo FastAPI. A instância
`_service` é um singleton criado no import e reutilizada pela dependência.
"""

from app.services.llm_service import LlmService

_service = LlmService()


def get_llm_service() -> LlmService:
    """Retorna a instância singleton de `LlmService` para injeção.

    Returns:
        LlmService: instância reutilizável do serviço de chat.
    """
    return _service
