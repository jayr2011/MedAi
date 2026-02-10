from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

CHROMA_DIR = Path("./chroma_db")
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

_embeddings = None
_vectorstore = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None and CHROMA_DIR.exists():
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=get_embeddings()
        )
    return _vectorstore

def ingest_pdf(file_path: str) -> dict:
    """Processa um PDF e salva os embeddings no ChromaDB"""
    global _vectorstore

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

    if _vectorstore is None:
        _vectorstore = Chroma.from_documents(
            chunks,
            get_embeddings(),
            persist_directory=str(CHROMA_DIR)
        )
    else:
        _vectorstore.add_documents(chunks)

    return {
        "file": file_name,
        "chunks": len(chunks),
        "pages": len(documents)
    }

def buscar_contexto(pergunta: str, k: int = 3) -> str:
    """Busca os chunks mais relevantes para a pergunta"""
    vs = get_vectorstore()
    if vs is None:
        return ""

    docs = vs.similarity_search(pergunta, k=k)
    if not docs:
        return ""

    return "\n\n---\n\n".join([
        f"[Fonte: {doc.metadata.get('source', '?')} | Pág. {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    ])


def listar_documentos() -> list[str]:
    """Lista os documentos já ingeridos"""
    vs = get_vectorstore()
    if vs is None:
        return []

    metadatas = vs.get()["metadatas"]
    return list(set(m.get("source", "?") for m in metadatas))

def deletar_documento(file_name: str) -> bool:
    """Deleta um documento do vectorstore"""
    vs = get_vectorstore()
    if vs is None:
        return False

    ids_to_delete = []
    data = vs.get()
    for i, meta in enumerate(data["metadatas"]):
        if meta.get("source") == file_name:
            ids_to_delete.append(data["ids"][i])

    if ids_to_delete:
        vs.delete(ids=ids_to_delete)
        return True
    return False