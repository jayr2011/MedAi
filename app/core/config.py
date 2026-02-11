from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    databricks_url: str
    databricks_token: str
    max_tokens: Optional[int] = None
    debug: bool = True
    huggingface_token: Optional[str] = None

    router_threshold: float = 0.5
    min_fallback_length: int = 50
    score_alert_band: float = 0.05
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
