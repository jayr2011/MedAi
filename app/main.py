import logging
from fastapi import FastAPI
from app.api.v1.endpoints import chat
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

@app.get("/health")
async def health():
    return {
        "status": "OK",
        "databricks_url": settings.databricks_url[:50] + "...",
        "max_tokens": settings.max_tokens
    }

FRONTEND_DIR = Path("/app/frontend") if Path("/app/frontend").exists() else Path(__file__).parent.parent / "frontend" / "dist"

# Monta o frontend apenas se o diretório existir
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")