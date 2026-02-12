"""Dependências compartilhadas do aplicativo.

Este módulo expõe factories/dependencies usadas pelo FastAPI. A instância
`_service` é um singleton criado no import e reutilizada pela dependência.
"""

from app.services.databricks_service import DatabricksService

_service = DatabricksService()


def get_databricks_service() -> DatabricksService:
    """Retorna a instância singleton de `DatabricksService` para injeção.

    Returns:
        DatabricksService: instância reutilizável do serviço Databricks.
    """
    return _service
