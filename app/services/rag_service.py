"""Serviço RAG para ingestão, armazenamento e busca de documentos PDF.

Este módulo fornece helpers singleton para embeddings e vectorstore (Chroma),
funções para ingestão de PDFs com chunking e metadata, e operações de
consulta/remoção sobre o vectorstore persistente.

O vectorstore é armazenado em disco no diretório especificado por CHROMA_DIR
e mantém persistência entre reinicializações da aplicação.

Attributes:
    CHROMA_DIR: Diretório onde o banco de dados Chroma é persistido.
    UPLOAD_DIR: Diretório onde os PDFs carregados são armazenados.

Example:
    Uso típico do serviço RAG:
    
    >>> # Ingerir um documento
    >>> result = ingest_pdf("./uploads/manual.pdf")
    >>> print(f"Processado {result['chunks']} chunks")
    
    >>> # Buscar contexto relevante
    >>> context = buscar_contexto("sintomas de diabetes", k=3)
    >>> print(context)
    
    >>> # Listar documentos
    >>> docs = listar_documentos()
    >>> print(docs)
"""

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
"""Path: Diretório de persistência do vectorstore Chroma."""

UPLOAD_DIR = Path("./uploads")
"""Path: Diretório onde arquivos PDF carregados são armazenados."""

UPLOAD_DIR.mkdir(exist_ok=True)

logger.info("RAG service inicializado. CHROMA_DIR=%s UPLOAD_DIR=%s", CHROMA_DIR, UPLOAD_DIR)

_embeddings = None
_vectorstore = None

def get_embeddings():
    """Retorna instância singleton do modelo de embeddings HuggingFace.
    
    Cria e cacheia uma instância do modelo de embeddings usando
    sentence-transformers/all-MiniLM-L6-v2. A instância é reutilizada
    em chamadas subsequentes para otimizar performance.

    Returns:
        HuggingFaceEmbeddings configurado para execução em CPU com
        normalização de embeddings habilitada.

    Note:
        A instância é criada apenas uma vez e armazenada globalmente.
        O modelo é executado em CPU e normaliza os embeddings para
        melhorar a qualidade das buscas por similaridade.
    """
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
    """Retorna instância singleton do Chroma vectorstore persistido.
    
    Tenta abrir um vectorstore Chroma existente do diretório de persistência.
    Se o diretório não existir ou houver erro na inicialização, retorna None.
    A instância é cacheada globalmente para reutilização.

    Returns:
        Instância do Chroma vectorstore se disponível, None caso contrário.
        None também é retornado se ocorrer erro durante a inicialização.

    Note:
        Se CHROMA_DIR não existir, assume que nenhum documento foi ingerido
        ainda. Em caso de erro de inicialização, detalhes são logados e
        None é retornado.
    """
    global _vectorstore
    if _vectorstore is None:
        if CHROMA_DIR.exists():
            logger.info("Inicializando Chroma a partir de %s", CHROMA_DIR)
            try:
                # Tenta carregar o vectorstore Chroma do diretório persistente
                # para preservar contexto entre reinicializações.
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
    """Ingere um PDF, divide em chunks e persiste no vectorstore.
    
    Processa o PDF especificado carregando seu conteúdo, dividindo em chunks
    com overlapping usando RecursiveCharacterTextSplitter, e armazenando no
    vectorstore Chroma. Cada chunk recebe metadata com o nome do arquivo fonte.
    
    Se o vectorstore ainda não existir, ele é criado. Caso contrário, os chunks
    são adicionados ao vectorstore existente.

    Args:
        file_path: Caminho completo para o arquivo PDF no filesystem.

    Returns:
        Dicionário contendo:
            - file (str): Nome do arquivo processado
            - chunks (int): Número de chunks criados
            - pages (int): Número de páginas no PDF

    Raises:
        Exception: Propaga qualquer erro ocorrido durante leitura do PDF,
            chunking ou persistência no vectorstore. Erros são logados antes
            de serem propagados.

    Example:
        >>> result = ingest_pdf("./uploads/manual_medico.pdf")
        >>> print(f"Processado: {result['file']}")
        Processado: manual_medico.pdf
        >>> print(f"Chunks: {result['chunks']}, Páginas: {result['pages']}")
        Chunks: 42, Páginas: 15
    """
    global _vectorstore

    logger.info("Iniciando ingestão do PDF %s", file_path)
    try:
        # Carrega o PDF e divide em chunks gerenciáveis
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
            # Adiciona metadado de origem para cada chunk
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
    """Busca e formata trechos relevantes do vectorstore por similaridade.
    
    Executa busca semântica no vectorstore usando similarity_search e retorna
    os documentos mais relevantes formatados com metadata (fonte e página).
    Os documentos são concatenados separados por delimitador visual.

    Args:
        pergunta: Texto da consulta para busca semântica.
        k: Número máximo de documentos a retornar. Padrão é 5.

    Returns:
        String contendo contexto concatenado e formatado. Cada documento inclui
        metadata de fonte e página, separados por "---". Retorna string vazia
        se vectorstore não estiver disponível, houver erro, ou não encontrar
        correspondências.

    Example:
        >>> context = buscar_contexto("sintomas de diabetes", k=3)
        >>> print(context)
        [Fonte: manual_medico.pdf | Pág. 15]
        Os sintomas incluem sede excessiva...
        
        ---
        
        [Fonte: manual_medico.pdf | Pág. 16]
        O diagnóstico é feito através...
    """
    vs = get_vectorstore()
    if vs is None:
        logger.debug("buscar_contexto: vectorstore não disponível.")
        return ""

    try:
        # Executa busca semântica por similaridade
        docs = vs.similarity_search(pergunta, k=k)
    except Exception as e:
        logger.exception("Erro ao executar similarity_search: %s", e)
        return ""

    if not docs:
        logger.debug("buscar_contexto: nenhuma correspondência encontrada para a pergunta.")
        return ""

    logger.debug("buscar_contexto: %d documentos retornados para pergunta '%s'", len(docs), pergunta)

    # Formata cada documento com fonte/página e concatena separando por ---
    return "\n\n---\n\n".join([
        f"[Fonte: {doc.metadata.get('source', '?')} | Pág. {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    ])

def listar_documentos() -> list[str]:
    """Retorna nomes únicos de documentos ingeridos no vectorstore.
    
    Extrai os nomes de arquivos únicos a partir dos metadados persistidos
    no vectorstore. Cada chunk possui metadata com campo 'source' contendo
    o nome do arquivo original.

    Returns:
        Lista de nomes de arquivos (strings) encontrados nos metadados.
        Retorna lista vazia se vectorstore não estiver disponível ou
        em caso de erro durante leitura.

    Example:
        >>> docs = listar_documentos()
        >>> print(docs)
        ['manual_medico.pdf', 'protocolo_2024.pdf']
    """
    vs = get_vectorstore()
    if vs is None:
        logger.debug("listar_documentos: vectorstore não disponível.")
        return []

    try:
        # Extrai os nomes dos arquivos a partir dos metadados persistidos
        metadatas = vs.get()["metadatas"]
        docs = list(set(m.get("source", "?") for m in metadatas))
        logger.info("listar_documentos: %d documentos listados", len(docs))
        return docs
    except Exception as e:
        logger.exception("Erro ao listar documentos: %s", e)
        return []
def deletar_documento(file_name: str) -> bool:
    """Remove do vectorstore todos os chunks associados a um documento.
    
    Busca todos os chunks cujo metadata contém source igual a file_name
    e os remove do vectorstore. A operação é permanente e afeta o banco
    de dados persistido.

    Args:
        file_name: Nome exato do arquivo cujos chunks devem ser removidos.
            Deve corresponder ao valor armazenado no campo 'source' dos
            metadados.

    Returns:
        True se ao menos um chunk foi deletado com sucesso, False se
        nenhum chunk foi encontrado ou se vectorstore não estiver disponível.

    Example:
        >>> success = deletar_documento("manual_medico.pdf")
        >>> if success:
        ...     print("Documento removido com sucesso")
        Documento removido com sucesso
    """
    logger.info("Solicitação para deletar documento %s", file_name)
    vs = get_vectorstore()
    if vs is None:
        logger.debug("deletar_documento: vectorstore não disponível.")
        return False

    try:
        # Recupera IDs correspondentes ao `source` indicado e deleta-os
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