from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.rag_service import (
    ingest_pdf_semantic,
    listar_documentos, 
    deletar_documento, 
    UPLOAD_DIR
)
import shutil
import logging

router = APIRouter(prefix="/rag", tags=["rag"])

logger = logging.getLogger(__name__)

@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas PDFs são suportados")

    file_path = UPLOAD_DIR / file.filename
    
    if file_path.exists():
        raise HTTPException(status_code=409, detail=f"Arquivo '{file.filename}' já existe")
    
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info("Arquivo salvo: %s", file_path)
        
        result = ingest_pdf_semantic(str(file_path))
        
        logger.info("Processamento concluído: %s", result)
        
        return {
            "status": "ok",
            "message": f"'{file.filename}' processado com sucesso",
            **result
        }
    except Exception as e:
        logger.exception("Erro ao processar '%s'", file.filename)
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")


@router.get("/documents")
async def list_documents():
    return {"documents": listar_documentos()}


@router.delete("/documents/{file_name}")
async def delete_document(file_name: str):
    if deletar_documento(file_name):
        file_path = UPLOAD_DIR / file_name
        if file_path.exists():
            file_path.unlink()
        return {"status": "ok", "message": f"'{file_name}' removido"}
    raise HTTPException(status_code=404, detail="Documento não encontrado")