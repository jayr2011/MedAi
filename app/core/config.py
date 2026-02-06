from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    databricks_url: str = ""
    databricks_token: str = ""
    max_tokens: int = 2048
    debug: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
