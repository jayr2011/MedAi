from pydantic import BaseModel
from typing import List, Optional

"""Pydantic schemas usados pelas rotas de chat.

Schemas:
- ChatMessage: representa uma única mensagem (role + content).
- ChatRequest: payload enviado ao endpoint de chat com lista de mensagens
  e histórico opcional.
"""


class ChatMessage(BaseModel):
    """Representa uma mensagem no diálogo.

    Attributes:
        role (str): papel da mensagem (ex.: 'user', 'assistant', 'system').
        content (str): texto da mensagem.
    """
    role: str
    content: str


class ChatRequest(BaseModel):
    """Payload de requisição para endpoints de chat.

    Attributes:
        messages (List[ChatMessage]): lista ordenada de mensagens (a última
            costuma ser a entrada do usuário atual).
        history (Optional[List[ChatMessage]]): mensagens históricas adicionais
            (padrão: lista vazia).
    """
    messages: List[ChatMessage]
    history: Optional[List[ChatMessage]] = []
