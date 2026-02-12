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
    """Health check endpoint.

    Returns:
        dict: informações básicas sobre o serviço e configurações relevantes
            (ex.: `databricks_url` truncada e `max_tokens`).
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