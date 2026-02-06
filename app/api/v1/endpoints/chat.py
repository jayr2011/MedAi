from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.api.v1.schemas.chat import ChatRequest
from app.services.databricks_service import DatabricksService
from app.api.deps import get_databricks_service

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: DatabricksService = Depends(get_databricks_service)
):
    """Endpoint principal: Proxy stream para Databricks LLM"""
    try:
        async def generate():
            async for chunk in service.chat_stream(request.messages):
                yield f"data: {chunk}\n\n"
        
        return StreamingResponse(
            generate(), 
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Databricks error: {str(e)}")