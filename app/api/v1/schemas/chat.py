from pydantic import BaseModel
from typing import List, Optional

"""Tipos de dados para mensagens de chat e requisições, utilizados para validação e documentação da API."""
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    history: Optional[List[ChatMessage]] = []
