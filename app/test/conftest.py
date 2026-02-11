import pytest
import asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.fixture(scope="session")
def event_loop():
    """Cria um loop de eventos para testes assíncronos."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

pytest.fixture()
async def ac() -> AsyncGenerator:
    """
    Retorna um cliente HTTP assíncrono para interagir com a API.
    """
    async with AsyncClient(transport=ASGITransport(app=app),
                            base_url="http://test") as client:
        yield client