"""Carregamento centralizado de configurações via Pydantic Settings.

Este módulo expõe a classe `Settings` que agrega variáveis de
ambiente/`.env` usadas pela aplicação e a instância `settings`.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Configurações da aplicação carregadas automaticamente.

    Campos importantes:
        databricks_url (str): URL do endpoint Databricks.
        databricks_token (str): token de autenticação para Databricks.
        max_tokens (Optional[int]): número máximo de tokens para geração.
        debug (bool): ativa modo de debug (afeta verificação TLS no cliente HTTP).
        huggingface_token (Optional[str]): token para HF (quando usado).

    Configuração de carregamento:
        - valores podem vir de variáveis de ambiente ou do arquivo `.env`.
    """
    databricks_url: str
    use_semantic_chunking: bool = True
    max_concurrent_ingestions: int = 1
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


# Instância global reutilizável
settings = Settings()
