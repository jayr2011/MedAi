"""Serviço RAG para ingestão, busca e manutenção de documentos no Chroma."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from google import genai
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

logger = logging.getLogger(__name__)

CHROMA_DIR = Path("./chroma_db")
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

_embeddings = None
_vectorstore = None
_lock = threading.Lock()


class GeminiEmbeddingsAdapter:
    """Adapter de embeddings Gemini compatível com a interface esperada pelo Chroma."""

    def __init__(self, model_name: str = "gemini-embedding-001") -> None:
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        max_batch_size = 100
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), max_batch_size):
            batch = texts[i : i + max_batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
            )
            embeddings = getattr(response, "embeddings", None) or []
            all_embeddings.extend(
                [emb.values for emb in embeddings if getattr(emb, "values", None)]
            )

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        if not text:
            return []
        docs = self.embed_documents([text])
        return docs[0] if docs else []


def get_embeddings() -> GeminiEmbeddingsAdapter:
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                _embeddings = GeminiEmbeddingsAdapter()
                logger.info("Embeddings inicializados com %s", _embeddings.model_name)
    return _embeddings


def get_vectorstore() -> Chroma | None:
    global _vectorstore
    if _vectorstore is None and CHROMA_DIR.exists():
        with _lock:
            if _vectorstore is None:
                try:
                    _vectorstore = Chroma(
                        persist_directory=str(CHROMA_DIR),
                        embedding_function=get_embeddings(),
                    )
                    logger.info("Chroma inicializado em %s", CHROMA_DIR)
                except Exception:
                    logger.exception("Erro ao inicializar Chroma")
                    _vectorstore = None
    return _vectorstore


def _load_pdf(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()


def _prepare_chunks(chunks: list[Document], file_name: str, method: str) -> int:
    total_chars = 0
    for idx, chunk in enumerate(chunks):
        chunk.metadata["source"] = file_name
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["chunk_method"] = method
        total_chars += len(chunk.page_content)
    return total_chars


def _persist_chunks(chunks: list[Document]) -> None:
    global _vectorstore

    if _vectorstore is None:
        _vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
    else:
        _vectorstore.add_documents(chunks)


def ingest_pdf(file_path: str) -> dict:
    """Ingere PDF usando chunking recursivo (compatibilidade retroativa)."""
    global _vectorstore

    documents = _load_pdf(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    file_name = Path(file_path).name
    _prepare_chunks(chunks, file_name=file_name, method="recursive")
    _persist_chunks(chunks)

    return {
        "file": file_name,
        "chunks": len(chunks),
        "pages": len(documents),
    }


def ingest_pdf_semantic(file_path: str) -> dict:
    """Ingere PDF com separadores semânticos básicos para melhor contexto."""
    global _vectorstore

    documents = _load_pdf(file_path)
    semantic_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=220,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = semantic_splitter.split_documents(documents)

    file_name = Path(file_path).name
    total_chars = _prepare_chunks(chunks, file_name=file_name, method="semantic")
    avg_chunk_size = total_chars // len(chunks) if chunks else 0

    _persist_chunks(chunks)

    return {
        "file": file_name,
        "chunks": len(chunks),
        "pages": len(documents),
        "avg_chunk_size": avg_chunk_size,
        "method": "semantic",
    }


def buscar_contexto(pergunta: str, k: int = 5) -> str:
    vs = get_vectorstore()
    if vs is None:
        return ""

    try:
        docs = vs.similarity_search(pergunta, k=k)
    except Exception:
        logger.warning("Erro ao executar similarity_search; seguindo sem contexto RAG.")
        return ""

    if not docs:
        return ""

    return "\n\n---\n\n".join(
        [
            f"[Fonte: {doc.metadata.get('source', '?')} | Pág. {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        ]
    )


def listar_documentos() -> list[str]:
    vs = get_vectorstore()
    if vs is None:
        return []

    try:
        data = vs.get()
        metadatas = data.get("metadatas", [])
        docs = sorted({m.get("source") for m in metadatas if m and m.get("source")})
        return docs
    except Exception:
        logger.exception("Erro ao listar documentos")
        return []


def deletar_documento(file_name: str) -> bool:
    vs = get_vectorstore()
    if vs is None:
        return False

    try:
        data = vs.get()
        metadatas = data.get("metadatas", [])
        ids = data.get("ids", [])

        ids_to_delete = [ids[i] for i, meta in enumerate(metadatas) if meta.get("source") == file_name]
        if not ids_to_delete:
            return False

        vs.delete(ids=ids_to_delete)
        return True
    except Exception:
        logger.exception("Erro ao deletar documento %s", file_name)
        return False
