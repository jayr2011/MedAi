import numpy as np
import app.services.web_search_service as wss

class FakeModel:
    """Modelo falso para simular o comportamento de embeddings sem depender de implementações reais."""
    def __init__(self, docs_emb=None, query_emb=None, raise_on_embed_documents=False):
        self._docs_emb = docs_emb
        self._query_emb = query_emb
        self._raise_on_embed_documents = raise_on_embed_documents

    def embed_documents(self, docs):
        if self._raise_on_embed_documents:
            raise RuntimeError("embed_documents should not be called")
        return self._docs_emb

    def embed_query(self, query):
        return self._query_emb

def test_returns_true_when_score_above_threshold(monkeypatch):
    """Testa se a função retorna True quando o score é acima do limiar."""
    wss._embeddings_cache = None
    docs_emb = [[1.0, 0.0], [0.0, 1.0]]
    query_emb = [1.0, 0.0]
    model = FakeModel(docs_emb=docs_emb, query_emb=query_emb)
    monkeypatch.setattr(wss, "get_embeddings", lambda: model)
    assert wss.deve_pesquisar_web("qualquer", threshold=0.9)

def test_returns_false_when_score_below_threshold(monkeypatch):
    """Testa se a função retorna False quando o score é abaixo do limiar."""
    wss._embeddings_cache = None
    docs_emb = [[1.0, 0.0], [1.0, 0.0]]
    query_emb = [0.0, 1.0]
    model = FakeModel(docs_emb=docs_emb, query_emb=query_emb)
    monkeypatch.setattr(wss, "get_embeddings", lambda: model)
    assert not wss.deve_pesquisar_web("outra", threshold=0.5)

def test_handles_exception_and_returns_false(monkeypatch):
    """Testa se a função lida com exceções e retorna False em caso de erro."""
    wss._embeddings_cache = None
    def raise_get_embeddings():
        raise RuntimeError("erro intencional")
    monkeypatch.setattr(wss, "get_embeddings", raise_get_embeddings)
    assert not wss.deve_pesquisar_web("erro")

def test_uses_embeddings_cache_when_present(monkeypatch):
    """Testa se a função utiliza o cache de embeddings quando disponível."""
    wss._embeddings_cache = np.array([[0.0, 1.0], [1.0, 0.0]])
    model = FakeModel(docs_emb=None, query_emb=[0.0, 1.0], raise_on_embed_documents=True)
    monkeypatch.setattr(wss, "get_embeddings", lambda: model)
    assert wss.deve_pesquisar_web("qualquer")