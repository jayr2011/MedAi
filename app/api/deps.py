from app.services.databricks_service import DatabricksService

def get_databricks_service() -> DatabricksService:
    return DatabricksService()
