"""Ponto de entrada da API FastAPI e configuração global da aplicação.

Configura middleware CORS, registra routers (v1/chat, v1/rag) e monta os
arquivos estáticos do frontend quando disponíveis.
"""

import logging
from fastapi import FastAPI
from app.api.v1.endpoints import chat
from app.api.v1.endpoints import rag
from app.core.config import settings
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/v1")
app.include_router(rag.router, prefix="/v1")

@app.get("/health")
async def health():
    """Verifica o status e configuração da aplicação.

    Endpoint de health check que retorna informações básicas sobre o serviço,
    incluindo status operacional, configuração do Databricks e parâmetros
    de geração de texto.

    Returns:
        Dicionário contendo:
            - status (str): Sempre "OK" quando o serviço está operacional
            - databricks_url (str): URL truncada do endpoint Databricks (primeiros 50 caracteres)
            - max_tokens (int): Limite máximo de tokens configurado para geração

    Example:
        >>> # GET /health
        >>> {
        ...     "status": "OK",
        ...     "databricks_url": "https://adb-1234567890123456.7.azuredatabricks.net...",
        ...     "max_tokens": 512
        ... }
    """ 
    return {
        "status": "OK",
        "databricks_url": settings.databricks_url[:50] + "...",
        "max_tokens": settings.max_tokens
    }

# Determina o diretório do build do frontend (prioriza `/app/frontend` em
# ambientes de container, caso contrário usa `frontend/dist` local).
FRONTEND_DIR = Path("/app/frontend") if Path("/app/frontend").exists() else Path(__file__).parent.parent / "frontend" / "dist"

# Se o frontend estiver presente, monta os arquivos estáticos na raiz
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")