from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    databricks_url: str = ""
    databricks_token: str = ""
    max_tokens: Optional[int] = None
    debug: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
