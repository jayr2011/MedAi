from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import logging
import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CHROMA_DIR = Path("./chroma_db")
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

logger.info("RAG service inicializado. CHROMA_DIR=%s UPLOAD_DIR=%s", CHROMA_DIR, UPLOAD_DIR)

_embeddings = None
_vectorstore = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Embeddings inicializados: model=sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        if CHROMA_DIR.exists():
            logger.info("Inicializando Chroma a partir de %s", CHROMA_DIR)
            try:
                _vectorstore = Chroma(
                    persist_directory=str(CHROMA_DIR),
                    embedding_function=get_embeddings()
                )
                logger.info("Chroma inicializado com persist_directory=%s", CHROMA_DIR)
            except Exception as e:
                logger.exception("Erro ao inicializar Chroma (persist_directory=%s): %s", CHROMA_DIR, e)
                _vectorstore = None
        else:
            logger.info("Chroma DB não encontrado em %s. Nenhum documento ingerido ainda.", CHROMA_DIR)
    return _vectorstore

def ingest_pdf(file_path: str) -> dict:
    """Processa um PDF e salva os embeddings no ChromaDB"""
    global _vectorstore

    logger.info("Iniciando ingestão do PDF %s", file_path)
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        file_name = Path(file_path).name
        for chunk in chunks:
            chunk.metadata["source"] = file_name

        logger.info("Processado %s: %d chunks em %d páginas", file_name, len(chunks), len(documents))

        if _vectorstore is None:
            _vectorstore = Chroma.from_documents(
                chunks,
                get_embeddings(),
                persist_directory=str(CHROMA_DIR)
            )
            logger.info("Chroma criado e persistido em %s com %d chunks", CHROMA_DIR, len(chunks))
        else:
            _vectorstore.add_documents(chunks)
            logger.info("Adicionados %d chunks ao vectorstore existente", len(chunks))

        return {
            "file": file_name,
            "chunks": len(chunks),
            "pages": len(documents)
        }
    except Exception as e:
        logger.exception("Erro ao ingerir PDF %s: %s", file_path, e)
        raise

def buscar_contexto(pergunta: str, k: int = 5) -> str:
    """Busca os chunks mais relevantes para a pergunta"""
    vs = get_vectorstore()
    if vs is None:
        logger.debug("buscar_contexto: vectorstore não disponível.")
        return ""

    try:
        docs = vs.similarity_search(pergunta, k=k)
    except Exception as e:
        logger.exception("Erro ao executar similarity_search: %s", e)
        return ""

    if not docs:
        logger.debug("buscar_contexto: nenhuma correspondência encontrada para a pergunta.")
        return ""

    logger.debug("buscar_contexto: %d documentos retornados para pergunta '%s'", len(docs), pergunta)

    return "\n\n---\n\n".join([
        f"[Fonte: {doc.metadata.get('source', '?')} | Pág. {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    ])


def listar_documentos() -> list[str]:
    """Lista os documentos já ingeridos"""
    vs = get_vectorstore()
    if vs is None:
        logger.debug("listar_documentos: vectorstore não disponível.")
        return []

    try:
        metadatas = vs.get()["metadatas"]
        docs = list(set(m.get("source", "?") for m in metadatas))
        logger.info("listar_documentos: %d documentos listados", len(docs))
        return docs
    except Exception as e:
        logger.exception("Erro ao listar documentos: %s", e)
        return []

def deletar_documento(file_name: str) -> bool:
    """Deleta um documento do vectorstore"""
    logger.info("Solicitação para deletar documento %s", file_name)
    vs = get_vectorstore()
    if vs is None:
        logger.debug("deletar_documento: vectorstore não disponível.")
        return False

    try:
        ids_to_delete = []
        data = vs.get()
        for i, meta in enumerate(data["metadatas"]):
            if meta.get("source") == file_name:
                ids_to_delete.append(data["ids"][i])

        if ids_to_delete:
            vs.delete(ids=ids_to_delete)
            logger.info("Documento %s deletado (%d chunks)", file_name, len(ids_to_delete))
            return True
        else:
            logger.debug("deletar_documento: nenhum chunk encontrado para %s", file_name)
            return False
    except Exception as e:
        logger.exception("Erro ao deletar documento %s: %s", file_name, e)
        return False