from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    databricks_url: str = ""
    databricks_token: str = ""
    max_tokens: Optional[int] = None
    debug: bool = True

    router_threshold: float = 0.5
    min_fallback_length: int = 50
    score_alert_band: float = 0.05
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
