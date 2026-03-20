import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from google.genai import types

from app.services.llm_service import LlmService
from app.api.v1.schemas.chat import ChatMessage


class TestLlmService:
    """Testes unitários para LlmService."""

    @patch('app.services.llm_service.genai.Client')
    @patch('app.services.llm_service.settings')
    def test_init_success(self, mock_settings, mock_client):
        """Testa inicialização bem-sucedida do LlmService."""
        mock_settings.gemini_api_key = "test-key"
        mock_settings.llm_model = "gemini-2.5-flash-lite"
        
        service = LlmService()
        
        mock_client.assert_called_once_with(api_key="test-key")
        assert service.client is not None
        assert service.model_name == "gemini-2.5-flash-lite"

    @patch('app.services.llm_service.genai.Client')
    @patch('app.services.llm_service.settings')
    def test_init_failure(self, mock_settings, mock_client):
        """Testa falha na inicialização do LlmService."""
        mock_client.side_effect = Exception("API error")
        mock_settings.llm_model = "gemini-2.5-flash-lite"
        
        service = LlmService()
        
        assert service.client is None
        assert service.model_name == "gemini-2.5-flash-lite"

    @patch('app.services.llm_service.buscar_contexto')
    async def test_coletar_contextos_success(self, mock_buscar):
        """Testa coleta de contextos com sucesso."""
        mock_buscar.return_value = "Contexto RAG de teste"
        service = LlmService()
        
        with patch.object(service, 'client', Mock()):
            contexto_rag, contexto_web = await service._coletar_contextos("pergunta teste")
        
        assert contexto_rag == "Contexto RAG de teste"
        assert contexto_web == ""

    @patch('app.services.llm_service.buscar_contexto')
    async def test_coletar_contextos_error(self, mock_buscar):
        """Testa coleta de contextos com erro."""
        mock_buscar.side_effect = Exception("RAG error")
        service = LlmService()
        
        with patch.object(service, 'client', Mock()):
            contexto_rag, contexto_web = await service._coletar_contextos("pergunta teste")
        
        assert contexto_rag == ""
        assert contexto_web == ""

    async def test_chat_stream_no_client(self):
        """Testa chat_stream quando cliente não está inicializado."""
        service = LlmService()
        service.client = None
        
        with pytest.raises(ValueError, match="Cliente Gemini não inicializado"):
            async for chunk in service.chat_stream("pergunta", []):
                pass

    @patch('app.services.llm_service.buscar_contexto')
    async def test_chat_stream_gemma_model(self, mock_buscar):
        """Testa chat_stream com modelo Gemma (sem system_instruction)."""
        mock_buscar.return_value = "Contexto teste"
        
        service = LlmService()
        service.model_name = "gemma-2b"
        
        mock_response = Mock()
        mock_chunk = Mock()
        mock_chunk.text = "Resposta teste"
        mock_response.__aiter__ = Mock(return_value=iter([mock_chunk]))
        
        mock_client = Mock()
        mock_client.models.generate_content_stream.return_value = mock_response
        service.client = mock_client
        
        history = [ChatMessage(role="user", content="pergunta anterior")]
        
        chunks = []
        async for chunk in service.chat_stream("pergunta atual", history):
            chunks.append(chunk)
        
        assert chunks == ["Resposta teste"]
        
        call_args = mock_client.models.generate_content_stream.call_args
        config = call_args.kwargs['config']
        assert config.system_instruction is None
        assert config.temperature == 0.2

    @patch('app.services.llm_service.buscar_contexto')
    async def test_chat_stream_gemini_model(self, mock_buscar):
        """Testa chat_stream com modelo Gemini (com system_instruction)."""
        mock_buscar.return_value = "Contexto RAG teste"
        
        service = LlmService()
        service.model_name = "gemini-2.5-flash-lite"
        
        mock_response = Mock()
        mock_chunk = Mock()
        mock_chunk.text = "Resposta teste"
        mock_response.__aiter__ = Mock(return_value=iter([mock_chunk]))
        
        mock_client = Mock()
        mock_client.models.generate_content_stream.return_value = mock_response
        service.client = mock_client
        
        history = [ChatMessage(role="user", content="pergunta anterior")]
        
        chunks = []
        async for chunk in service.chat_stream("pergunta atual", history):
            chunks.append(chunk)
        
        assert chunks == ["Resposta teste"]
        
        call_args = mock_client.models.generate_content_stream.call_args
        config = call_args.kwargs['config']
        assert "MedAi" in config.system_instruction
        assert "Contexto RAG teste" in config.system_instruction
        assert config.temperature == 0.2

    @patch('app.services.llm_service.buscar_contexto')
    async def test_chat_stream_with_context(self, mock_buscar):
        """Testa chat_stream com contexto RAG e web."""
        mock_buscar.return_value = "Contexto RAG detalhado"
        
        service = LlmService()
        service.model_name = "gemini-2.5-flash-lite"
        
        mock_response = Mock()
        mock_chunk = Mock()
        mock_chunk.text = "Resposta com contexto"
        mock_response.__aiter__ = Mock(return_value=iter([mock_chunk]))
        
        mock_client = Mock()
        mock_client.models.generate_content_stream.return_value = mock_response
        service.client = mock_client
        
        chunks = []
        async for chunk in service.chat_stream("pergunta médica", []):
            chunks.append(chunk)
        
        assert chunks == ["Resposta com contexto"]
        
        call_args = mock_client.models.generate_content_stream.call_args
        config = call_args.kwargs['config']
        assert "Contexto RAG detalhado" in config.system_instruction
        assert "priorize os documentos locais" in config.system_instruction

    @patch('app.services.llm_service.buscar_contexto')
    async def test_chat_stream_error_handling(self, mock_buscar):
        """Testa tratamento de erros no chat_stream."""
        mock_buscar.return_value = "Contexto teste"
        
        service = LlmService()
        service.model_name = "gemini-2.5-flash-lite"
        
        mock_client = Mock()
        mock_client.models.generate_content_stream.side_effect = Exception("API Error")
        service.client = mock_client
        
        with pytest.raises(ValueError, match="Erro ao consultar provedor LLM"):
            async for chunk in service.chat_stream("pergunta", []):
                pass

    @patch('app.services.llm_service.buscar_contexto')
    async def test_chat_stream_history_formatting(self, mock_buscar):
        """Testa formatação correta do histórico de mensagens."""
        mock_buscar.return_value = ""
        
        service = LlmService()
        service.model_name = "gemini-2.5-flash-lite"
        
        mock_response = Mock()
        mock_chunk = Mock()
        mock_chunk.text = "Resposta"
        mock_response.__aiter__ = Mock(return_value=iter([mock_chunk]))
        
        mock_client = Mock()
        mock_client.models.generate_content_stream.return_value = mock_response
        service.client = mock_client
        
        history = [
            ChatMessage(role="user", content="pergunta 1"),
            ChatMessage(role="assistant", content="resposta 1"),
            ChatMessage(role="user", content="pergunta 2")
        ]
        
        chunks = []
        async for chunk in service.chat_stream("pergunta atual", history):
            chunks.append(chunk)
        
        call_args = mock_client.models.generate_content_stream.call_args
        contents = call_args.kwargs['contents']
        
        assert len(contents) == 5  # 3 do histórico + 1 pergunta atual
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "user"
        assert contents[3].role == "user"  # system instruction
        assert contents[4].role == "user"  # pergunta atual

    @patch('app.services.llm_service.buscar_contexto')
    async def test_chat_stream_empty_chunks(self, mock_buscar):
        """Testa chat_stream com chunks vazios."""
        mock_buscar.return_value = ""
        
        service = LlmService()
        service.model_name = "gemini-2.5-flash-lite"
        
        mock_response = Mock()
        mock_chunk_empty = Mock()
        mock_chunk_empty.text = None
        mock_chunk_with_text = Mock()
        mock_chunk_with_text.text = "Texto válido"
        mock_response.__aiter__ = Mock(return_value=iter([mock_chunk_empty, mock_chunk_with_text]))
        
        mock_client = Mock()
        mock_client.models.generate_content_stream.return_value = mock_response
        service.client = mock_client
        
        chunks = []
        async for chunk in service.chat_stream("pergunta", []):
            chunks.append(chunk)
        
        assert chunks == ["Texto válido"]
