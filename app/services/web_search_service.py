import logging
import numpy as np
from ddgs import DDGS
from sklearn.metrics.pairwise import cosine_similarity
from app.services.rag_service import get_embeddings
from app.core.config import settings

logger = logging.getLogger(__name__)

SCORE_ALERT_BAND = settings.score_alert_band
MIN_FALLBACK_LENGTH = settings.min_fallback_length

DOMINIOS_CONFIAVEIS = [
    ".gov.br", ".org.br", ".edu.br", "scielo.br", "pubmed.ncbi",
    "who.int", "paho.org", "msdmanuals.com", "einstein.br",
    "siriolibanes.br", "fleury.com.br", "pebmed.com.br", "medscape.com"
]

DOMINIOS_BLOQUEADOS = [
    "facebook.com", "instagram.com", "twitter.com", "tiktok.com",
    "youtube.com", "reddit.com", "quora.com", "yahoo.com",
    "reclameaqui.com.br", "mercadolivre.com.br", "shopee.com.br",
    "wikipedia.org"
]

EXEMPLOS_BUSCA = [
    "protocolos e diretrizes clínicas atualizados",
    "manejo de emergência e suporte avançado",
    "diretrizes para manejo de sepse e AVC",
    "manejo de cetoacidose e emergências metabólicas",
    "diagnóstico diferencial de sintomas comuns",
    "interações medicamentosas e ajustes de dose",
    "escores de risco e critérios de triagem"
]

_embeddings_cache = None

def deve_pesquisar_web(query: str, threshold: float | None = None) -> bool:
    """
    Decide semanticamente se a query precisa de busca externa comparando
    a pergunta com exemplos de temas médicos conhecidos.
    """
    global _embeddings_cache
    try:
        model = get_embeddings()
        if _embeddings_cache is None:
            docs_emb = model.embed_documents(EXEMPLOS_BUSCA)
            _embeddings_cache = np.array(docs_emb)

        query_vec = np.array(model.embed_query(query))
        scores = cosine_similarity([query_vec], _embeddings_cache)[0]
        score_max = np.max(scores)

        threshold_resolved = threshold if threshold is not None else settings.router_threshold
        
        logger.info(f"🧠 ROUTER: Score {score_max:.3f} (Limiar: {threshold_resolved})")
        return score_max >= threshold_resolved
    except Exception as e:
        logger.error(f"Erro no roteamento semântico: {e}")
        return False

def web_search(query: str, max_results=10) -> str:
    """
    Realiza a busca no DuckDuckGo, aplica filtros de domínios médicos
    e retorna o contexto formatado.
    """
    try:
        logger.info(f"🔎 Iniciando busca web para: {query}")
        with DDGS(verify=False) as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results, backend="brave"))
        
        if not raw_results:
            return ""

        resultados_filtrados = []
        for res in raw_results:
            link = res.get('href', '').lower()
            
            if any(bad in link for bad in DOMINIOS_BLOQUEADOS):
                continue
            
            if any(good in link for good in DOMINIOS_CONFIAVEIS):
                resultados_filtrados.append(res)

            if len(resultados_filtrados) >= max_results:
                break

        if not resultados_filtrados:
            for res in raw_results[:2]:
                if not any(bad in res.get('href', '') for bad in DOMINIOS_BLOQUEADOS):
                    resultados_filtrados.append(res)
        
        contexto = []
        for res in resultados_filtrados:
            contexto.append(
                f"- Fonte: {res.get('title')}\n  URL: {res.get('href')}\n  Resumo: {res.get('body')}"
            )
        return "\n\n".join(contexto)
    except Exception as e:
        logger.error(f"Erro na busca web: {e}")
        return ""