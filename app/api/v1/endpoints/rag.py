"""Endpoints RAG para ingestão, listagem e remoção de documentos PDF.

Este módulo fornece rotas FastAPI para gerenciar documentos no sistema RAG,
incluindo upload de PDFs, listagem de documentos ingeridos e remoção de
documentos do vectorstore.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.rag_service import (
    ingest_pdf, listar_documentos, deletar_documento, UPLOAD_DIR
)
import shutil

router = APIRouter(prefix="/rag", tags=["rag"])

@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Recebe um PDF via upload, persiste localmente e ingere no RAG.
    
    Processa o arquivo PDF enviado, divide em chunks e armazena no vectorstore
    para consultas posteriores via busca semântica. O arquivo original é mantido
    no diretório de uploads.

    Args:
        file: Arquivo PDF enviado pelo cliente via multipart/form-data.

    Returns:
        Dicionário contendo:
            - status (str): Sempre "ok" em caso de sucesso
            - message (str): Mensagem de confirmação com nome do arquivo
            - file (str): Nome do arquivo processado
            - chunks (int): Número de chunks criados
            - pages (int): Número de páginas processadas

    Raises:
        HTTPException: 
            - 400 se o arquivo não tiver extensão .pdf
            - 500 se houver erro durante processamento ou ingestão
    
    Example:
        >>> # POST /v1/rag/ingest
        >>> # Content-Type: multipart/form-data
        >>> # file: manual_medico.pdf
        >>> {
        ...     "status": "ok",
        ...     "message": "'manual_medico.pdf' processado com sucesso",
        ...     "file": "manual_medico.pdf",
        ...     "chunks": 42,
        ...     "pages": 15
        ... }
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas PDFs são suportados")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Processa o PDF e injeta no vectorstore via `ingest_pdf`.
        result = ingest_pdf(str(file_path))
        return {
            "status": "ok",
            "message": f"'{file.filename}' processado com sucesso",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")


@router.get("/documents")
async def list_documents():
    """Lista todos os documentos atualmente ingeridos no vectorstore.
    
    Retorna os nomes únicos dos arquivos que foram processados e estão
    disponíveis para consulta via busca semântica.

    Returns:
        Dicionário contendo:
            - documents (list[str]): Lista de nomes de arquivos ingeridos
    
    Example:
        >>> # GET /v1/rag/documents
        >>> {
        ...     "documents": [
        ...         "manual_medico.pdf",
        ...         "protocolo_2024.pdf"
        ...     ]
        ... }
    """
    return {"documents": listar_documentos()}


@router.delete("/documents/{file_name}")
async def delete_document(file_name: str):
    """Remove um documento do vectorstore e apaga o arquivo de upload.
    
    Deleta todos os chunks associados ao documento especificado do vectorstore
    e remove o arquivo original do diretório de uploads, se existir.

    Args:
        file_name: Nome do arquivo a ser removido (deve corresponder exatamente
            ao nome usado durante o upload).

    Returns:
        Dicionário contendo:
            - status (str): Sempre "ok" em caso de sucesso
            - message (str): Mensagem de confirmação com nome do arquivo

    Raises:
        HTTPException: 404 se o documento não for encontrado no vectorstore.
    
    Example:
        >>> # DELETE /v1/rag/documents/manual_medico.pdf
        >>> {
        ...     "status": "ok",
        ...     "message": "'manual_medico.pdf' removido"
        ... }
    """
    if deletar_documento(file_name):
        file_path = UPLOAD_DIR / file_name
        if file_path.exists():
            file_path.unlink()
        return {"status": "ok", "message": f"'{file_name}' removido"}
    raise HTTPException(status_code=404, detail="Documento não encontrado")