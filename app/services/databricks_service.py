import httpx
from typing import AsyncGenerator, List, Any
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
        payload = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "max_tokens": getattr(settings, 'max_tokens', 1024),
            "temperature": 0.5,
            "top_p": 0.7,
            "stream": True
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

    async def chat_sync(self, messages: List[ChatMessage]) -> dict[str, Any]:
        payload = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "max_tokens": 200,
            "temperature": 0.7
        }
        resp = await self.client.post(self.endpoint_url, json=payload)
        resp.raise_for_status()
        return resp.json()