from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.rag_service import (
    ingest_pdf, listar_documentos, deletar_documento, UPLOAD_DIR
)
import shutil

router = APIRouter(prefix="/rag", tags=["rag"])

@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload e ingestão de PDF para RAG"""
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas PDFs são suportados")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        """Processa o PDF e extrai informações para RAG"""
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
    """Lista documentos ingeridos"""
    return {"documents": listar_documentos()}


@router.delete("/documents/{file_name}")
async def delete_document(file_name: str):
    """Remove um documento do RAG"""
    if deletar_documento(file_name):
        file_path = UPLOAD_DIR / file_name
        if file_path.exists():
            file_path.unlink()
        return {"status": "ok", "message": f"'{file_name}' removido"}
    raise HTTPException(status_code=404, detail="Documento não encontrado")