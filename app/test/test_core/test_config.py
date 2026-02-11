from pydantic import ValidationError
from app.core.config import Settings
from unittest.mock import patch
import pytest

def test_settings_default_values():
    """Testa se os valores padrão estão corretos quando o ambiente está vazio."""
    mock_env = {
        "DATABRICKS_URL": "http://mock.ai",
        "DATABRICKS_TOKEN": "dapi_mock_123",
    }
    with patch.dict("os.environ", mock_env, clear=True):
        s = Settings(_env_file=None)
        assert s.databricks_url == "http://mock.ai"
        assert s.databricks_token == "dapi_mock_123"
        assert s.max_tokens is None
        assert s.debug is True
        assert s.huggingface_token is None

def test_settings_env_override():
    """Testa se a variável de ambiente sobrescreve o valor padrão."""
    mock_env = {
        "DATABRICKS_URL": "https://test.databricks.com",
        "DATABRICKS_TOKEN": "dapi_test_456",
        "MAX_TOKENS": "1000",
        "HUGGINGFACE_TOKEN": "hf_test_789"
    }
    with patch.dict("os.environ", mock_env, clear=True):
        s = Settings()
        assert s.databricks_url == "https://test.databricks.com"
        assert s.databricks_token == "dapi_test_456"
        assert s.max_tokens == 1000
        assert s.huggingface_token == "hf_test_789"

def test_settings_validation_error():
    """Testa se valores inválidos lançam erro de validação (ex: string em float)."""
    
    with patch.dict("os.environ", {"router_threshold": "not-a-float"}):
        with pytest.raises(ValidationError):
            Settings()
    with patch.dict("os.environ", {"min_fallback_length": "not-an-int"}):
        with pytest.raises(ValidationError):
            Settings()
    with patch.dict("os.environ", {"score_alert_band": "not-a-float"}):
        with pytest.raises(ValidationError):
            Settings()

def test_missing_required_settings():
    """Testa se a ausência de variáveis obrigatórias lança erro de validação."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValidationError):
            Settings(_env_file=None)