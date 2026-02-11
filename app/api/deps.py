from app.services.databricks_service import DatabricksService

_service = DatabricksService()

def get_databricks_service() -> DatabricksService:
    """Função de dependência para obter uma instância do DatabricksService."""
    return _service
