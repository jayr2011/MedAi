from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    gemini_api_key: str
    llm_model: str = "gemini-2.5-flash"
    use_semantic_chunking: bool = True
    max_concurrent_ingestions: int = 1
    max_tokens: Optional[int] = 1024
    debug: bool = True
    allowed_origins: list[str] = ["*"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
