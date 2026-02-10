import httpx
from typing import AsyncGenerator, List
from app.api.v1.schemas.chat import ChatMessage
from app.core.config import settings


class DatabricksService:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {settings.databricks_token}",
                "Content-Type": "application/json"
            },
            timeout=300.0,
            verify=not settings.debug
        )
        self.endpoint_url = settings.databricks_url

    async def chat_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        max_tokens = settings.max_tokens or 1024
        payload = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": True,
        }

        async with self.client.stream("POST", self.endpoint_url, json=payload) as response:
            if response.status_code != 200:
                error = await response.aread()
                raise ValueError(f"Databricks {response.status_code}: {error.decode()}")

            async for line in response.aiter_lines():
                stripped = line.strip()
                if stripped.startswith("data: "):
                    data = stripped[6:]
                    if data and data != "[DONE]":
                        yield data