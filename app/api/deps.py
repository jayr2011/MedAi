from app.services.databricks_service import DatabricksService

_service = DatabricksService()

def get_databricks_service() -> DatabricksService:
    return _service
