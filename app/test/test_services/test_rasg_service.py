from pathlib import Path
from app.services import rag_service
from unittest.mock import patch, MagicMock
import tempfile
import pytest

@pytest.fixture(autouse=True)
def reset_rag_state():
    """Garante que o estado global do rag_service seja resetado antes de cada teste."""
    old_e, old_vs = getattr(rag_service, "_embeddings"), getattr(rag_service, "_vectorstore")
    rag_service._embeddings = None
    rag_service._vectorstore = None
    yield
    rag_service._embeddings = old_e
    rag_service._vectorstore = old_vs


def test_get_embeddings_initialization():
    """Testa se os embeddings são inicializados corretamente e retornados na chamada subsequente."""
    mock_embeddings = MagicMock()
    with patch.object(rag_service, "_embeddings", None):
        with patch("app.services.rag_service.HuggingFaceEmbeddings", return_value=mock_embeddings) as MockEmb:
            e1 = rag_service.get_embeddings()
            assert e1 is mock_embeddings

            e2 = rag_service.get_embeddings()
            MockEmb.assert_called_once()
            assert e2 is e1

def test_get_embeddings_contructor_args():
    """Testa se os embeddings são inicializados com os argumentos corretos."""
    with patch.object(rag_service, "_embeddings", None):
        with patch("app.services.rag_service.HuggingFaceEmbeddings") as MockEmb:
            rag_service.get_embeddings()
            MockEmb.assert_called_once_with(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

def test_get_embeddings_on_failure():
    """Testa se falhas na inicialização dos embeddings são tratadas e não deixam o estado inconsistente."""
    with patch.object(rag_service, "_embeddings", None), \
         patch.object(rag_service, "HuggingFaceEmbeddings") as MockC:
        rag_service.get_embeddings()
        MockC.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )


def test_get_vectorstore_initialization():
    """Testa se o vectorstore é inicializado corretamente quando o diretório existe."""
    with patch("app.services.rag_service._vectorstore", None), \
         patch("app.services.rag_service._embeddings", None), \
         patch("app.services.rag_service.get_embeddings") as mock_get_embeddings, \
         patch("app.services.rag_service.Chroma") as MockChroma:
        
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()
        MockChroma.return_value = mock_vectorstore
        mock_get_embeddings.return_value = mock_embeddings

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("app.services.rag_service.CHROMA_DIR", Path(temp_dir)):
                vectorstore = rag_service.get_vectorstore()

        MockChroma.assert_called_once_with(
            persist_directory=str(Path(temp_dir)),
            embedding_function=mock_embeddings
        )
        assert vectorstore == mock_vectorstore

def test_get_vectorstore_no_directory():
    """Testa se o vectorstore retorna None quando o diretório não existe."""
    with patch("app.services.rag_service._vectorstore", None), \
         patch("app.services.rag_service.CHROMA_DIR", Path("/non/existent/dir")):
        vectorstore = rag_service.get_vectorstore()
        assert vectorstore is None

def test_get_vectorstore_initialization_failure():
    """Se Chroma levantar exceção durante a inicialização, retorna None e não altera _vectorstore."""
    mock_embeddings = MagicMock()
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(rag_service, "_vectorstore", None), \
             patch.object(rag_service, "_embeddings", None), \
             patch.object(rag_service, "CHROMA_DIR", Path(temp_dir)), \
             patch.object(rag_service, "get_embeddings", return_value=mock_embeddings), \
             patch.object(rag_service, "Chroma", side_effect=Exception("init failed")):
            vs = rag_service.get_vectorstore()
            assert vs is None
            assert getattr(rag_service, "_vectorstore") is None

def test_get_vectorstore_initialization_success():
    """Inicialização bem sucedida usando Chroma (mock)."""
    mock_vectorstore = MagicMock()
    mock_embeddings = MagicMock()
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(rag_service, "_vectorstore", None), \
             patch.object(rag_service, "_embeddings", None), \
             patch.object(rag_service, "CHROMA_DIR", Path(temp_dir)), \
             patch.object(rag_service, "get_embeddings", return_value=mock_embeddings), \
             patch.object(rag_service, "Chroma", return_value=mock_vectorstore) as MockChroma:
            vs = rag_service.get_vectorstore()
    MockChroma.assert_called_once_with(
        persist_directory=str(Path(temp_dir)),
        embedding_function=mock_embeddings
    )
    assert vs == mock_vectorstore

def test_ingest_pdf_inicialization():
    """Testa o processo de ingestão de um PDF, incluindo carregamento, divisão em chunks e armazenamento no vectorstore."""    
    mock_vectorstore = MagicMock()
    mock_embeddings = MagicMock()

    documents = [MagicMock(), MagicMock()]
    chunk1 = MagicMock(); chunk1.metadata = {}; chunk1.page_content = "c1"
    chunk2 = MagicMock(); chunk2.metadata = {}; chunk2.page_content = "c2"
    chunks = [chunk1, chunk2]

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "test.pdf"
        pdf_path.touch()

        with patch.object(rag_service, "_vectorstore", None), \
             patch.object(rag_service, "get_embeddings", return_value=mock_embeddings), \
             patch.object(rag_service, "CHROMA_DIR", Path(temp_dir)), \
             patch("app.services.rag_service.PyPDFLoader") as MockLoader, \
             patch("app.services.rag_service.RecursiveCharacterTextSplitter") as MockSplitter, \
             patch("app.services.rag_service.Chroma.from_documents", return_value=mock_vectorstore) as MockFromDocs:

            MockLoader.return_value.load.return_value = documents
            MockSplitter.return_value.split_documents.return_value = chunks

            result = rag_service.ingest_pdf(str(pdf_path))


            assert rag_service._vectorstore == mock_vectorstore

    MockFromDocs.assert_called_once_with(chunks, mock_embeddings, persist_directory=str(Path(temp_dir)))
    assert result["file"] == "test.pdf"
    assert result["chunks"] == len(chunks)
    assert result["pages"] == len(documents)

def test_ingest_pdf_failure():
    """Se Chroma.from_documents falhar, ingest_pdf deve propagar o erro e não alterar _vectorstore."""
    mock_embeddings = MagicMock()

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "test.pdf"
        pdf_path.touch()

        with patch.object(rag_service, "_vectorstore", None), \
             patch.object(rag_service, "get_embeddings", return_value=mock_embeddings), \
             patch.object(rag_service, "CHROMA_DIR", Path(temp_dir)), \
             patch("app.services.rag_service.PyPDFLoader") as MockLoader, \
             patch("app.services.rag_service.RecursiveCharacterTextSplitter") as MockSplitter, \
             patch("app.services.rag_service.Chroma.from_documents", side_effect=Exception("persist failed")):

            MockLoader.return_value.load.return_value = [MagicMock()]  # 1 página
            MockSplitter.return_value.split_documents.return_value = [MagicMock()]  # 1 chunk

            import pytest
            with pytest.raises(Exception, match="persist failed"):
                rag_service.ingest_pdf(str(pdf_path))

            assert getattr(rag_service, "_vectorstore") is None

def test_ingest_pdf_adds_to_existing_vectorstore():
    mock_vs = MagicMock()
    documents = [MagicMock()]
    chunk = MagicMock(); chunk.metadata = {}; chunk.page_content = "c"
    chunks = [chunk]

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "test.pdf"
        pdf_path.touch()

        with patch.object(rag_service, "_vectorstore", mock_vs), \
             patch("app.services.rag_service.PyPDFLoader") as MockLoader, \
             patch("app.services.rag_service.RecursiveCharacterTextSplitter") as MockSplitter, \
             patch("app.services.rag_service.Chroma.from_documents") as MockFromDocs:

            MockLoader.return_value.load.return_value = documents
            MockSplitter.return_value.split_documents.return_value = chunks

            res = rag_service.ingest_pdf(str(pdf_path))

            mock_vs.add_documents.assert_called_once_with(chunks)
            MockFromDocs.assert_not_called()
            assert res["chunks"] == len(chunks)

def test_buscar_contexto_no_vectorstore():
    """Testa se buscar_contexto retorna string vazia quando o vectorstore não está inicializado."""
    with patch.object(rag_service, "get_vectorstore", return_value=None):
        assert rag_service.buscar_contexto("qualquer") == ""

def test_buscar_contexto_similarity_exception():
    """Testa se buscar_contexto retorna string vazia quando similarity_search levanta exceção."""
    mock_vs = MagicMock()
    mock_vs.similarity_search.side_effect = Exception("boom")
    with patch.object(rag_service, "get_vectorstore", return_value=mock_vs):
        assert rag_service.buscar_contexto("pergunta") == ""

def test_buscar_contexto_empty_results():
    """Testa se buscar_contexto retorna string vazia quando não há documentos retornados."""
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = []
    with patch.object(rag_service, "get_vectorstore", return_value=mock_vs):
        assert rag_service.buscar_contexto("pergunta") == ""

def test_buscar_contexto_returns_formatted_text():
    """Testa se buscar_contexto retorna o texto formatado corretamente dos documentos retornados."""
    doc1 = MagicMock(); doc1.metadata = {"source": "a.pdf", "page": 1}; doc1.page_content = "conteudo A"
    doc2 = MagicMock(); doc2.metadata = {"source": "b.pdf", "page": 2}; doc2.page_content = "conteudo B"
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [doc1, doc2]

    with patch.object(rag_service, "get_vectorstore", return_value=mock_vs):
        out = rag_service.buscar_contexto("quem", k=2)
    mock_vs.similarity_search.assert_called_once_with("quem", k=2)

    expected = "[Fonte: a.pdf | Pág. 1]\nconteudo A\n\n---\n\n[Fonte: b.pdf | Pág. 2]\nconteudo B"
    assert out == expected

def test_listar_documentos_no_vectorstore():
    """Testa se listar_documentos retorna lista vazia quando o vectorstore não está inicializado."""
    with patch.object(rag_service, "get_vectorstore", return_value=None):
        assert rag_service.listar_documentos() == []

def test_listar_documentos_empty_metadatas():
    """Testa se listar_documentos retorna lista vazia quando o vectorstore retorna metadatas vazias."""
    mock_vs = MagicMock()
    mock_vs.get.return_value = {"metadatas": [], "ids": []}
    with patch.object(rag_service, "get_vectorstore", return_value=mock_vs):
        assert rag_service.listar_documentos() == []

def test_listar_documentos_returns_unique_sources():
    """Testa se listar_documentos retorna a lista de fontes únicas dos metadados."""
    mock_vs = MagicMock()
    mock_vs.get.return_value = {
        "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}, {"source": "a.pdf"}],
        "ids": ["1", "2", "3"]
    }
    with patch.object(rag_service, "get_vectorstore", return_value=mock_vs):
        result = rag_service.listar_documentos()
    assert set(result) == {"a.pdf", "b.pdf"}

def test_listar_documentos_handles_exception():
    """Testa se listar_documentos retorna lista vazia quando get() levanta exceção."""
    mock_vs = MagicMock()
    mock_vs.get.side_effect = Exception("boom")
    with patch.object(rag_service, "get_vectorstore", return_value=mock_vs):
        assert rag_service.listar_documentos() == []

def test_deletar_documento_no_vectorstore():
    """Testa se deletar_documento retorna False quando o vectorstore não está inicializado."""
    with patch.object(rag_service, "get_vectorstore", return_value=None):
        assert rag_service.deletar_documento("qualquer.pdf") is False

def test_deletar_documento_success_single_id():
    """Testa se deletar_documento chama delete() com o ID correto quando há uma correspondência única."""
    mock_vs = MagicMock()
    mock_vs.get.return_value = {
        "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}],
        "ids": ["id1", "id2"]
    }
    with patch.object(rag_service, "get_vectorstore", return_value=mock_vs):
        ok = rag_service.deletar_documento("a.pdf")
    mock_vs.delete.assert_called_once_with(ids=["id1"])
    assert ok is True

def test_deletar_documento_success_multiple_ids():
    """Testa se deletar_documento chama delete() com os IDs corretos quando há múltiplas correspondências."""
    mock_vs = MagicMock()
    mock_vs.get.return_value = {
        "metadatas": [{"source": "a.pdf"}, {"source": "a.pdf"}, {"source": "b.pdf"}],
        "ids": ["id1", "id2", "id3"]
    }
    with patch.object(rag_service, "get_vectorstore", return_value=mock_vs):
        ok = rag_service.deletar_documento("a.pdf")
    mock_vs.delete.assert_called_once_with(ids=["id1", "id2"])
    assert ok is True

def test_deletar_documento_no_matches_and_exception():
    """Testa se deletar_documento retorna False quando não há correspondências e também quando get() levanta exceção."""
    mock_vs = MagicMock()
    mock_vs.get.return_value = {"metadatas": [], "ids": []}
    with patch.object(rag_service, "get_vectorstore", return_value=mock_vs):
        assert rag_service.deletar_documento("x.pdf") is False
    mock_vs2 = MagicMock()
    mock_vs2.get.side_effect = Exception("boom")
    with patch.object(rag_service, "get_vectorstore", return_value=mock_vs2):
        assert rag_service.deletar_documento("x.pdf") is False