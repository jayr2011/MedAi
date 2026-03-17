"""Carregamento centralizado de configurações via Pydantic Settings.

Este módulo expõe a classe `Settings` que agrega variáveis de
ambiente/`.env` usadas pela aplicação e a instância `settings`.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Configurações da aplicação carregadas automaticamente.

    Campos importantes:
        llm_base_url (str): URL base do provedor OpenAI-compatible.
        llm_api_key (str): token/chave de autenticação do provedor LLM.
        llm_model (str): identificador do modelo/rota usado no chat.
        max_tokens (Optional[int]): número máximo de tokens para geração.
        debug (bool): ativa modo de debug (afeta verificação TLS no cliente HTTP).
        huggingface_token (Optional[str]): token para HF (quando usado).

    Configuração de carregamento:
        - valores podem vir de variáveis de ambiente ou do arquivo `.env`.
    """
    llm_base_url: str
    use_semantic_chunking: bool = True
    max_concurrent_ingestions: int = 1
    llm_api_key: str
    llm_model: str = "meta-llama-3-3-70b-instruct"
    llm_repo_id: str = "MaziyarPanahi/BioMistral-Clinical-7B-GGUF"
    llm_filename: str = "BioMistral-Clinical-7B.Q5_K_M.gguf"
    llm_threads: int = 4
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
