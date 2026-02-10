import logging
from ddgs import DDGS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.services.rag_service import get_embeddings
import json

logger = logging.getLogger(__name__)

from app.core.config import settings

SCORE_ALERT_BAND = settings.score_alert_band
MIN_FALLBACK_LENGTH = settings.min_fallback_length


def _get_router_threshold(explicit_threshold: float | None) -> float:
    """Resolve the threshold: explicit param overrides settings."""
    return explicit_threshold if explicit_threshold is not None else settings.router_threshold

DOMINIOS_CONFIAVEIS = [
    ".gov.br",
    ".org.br",
    ".edu.br",
    "scielo.br",
    "pubmed.ncbi",
    "who.int",
    "paho.org",
    "msdmanuals.com",
    "einstein.br",
    "siriolibanes.br",
    "fleury.com.br",
    "pebmed.com.br",
    "medscape.com"
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
    "causas menos prováveis para apresentações clínicas comuns",
    "avaliação de sintomas respiratórios e neurológicos agudos",
    "interações medicamentosas e ajustes de dose",
    "ajuste de dose em insuficiência renal ou hepática",
    "dosagem pediátrica e segurança de medicamentos",
    "escores de risco e critérios de triagem",
    "escores de gravidade e escalas de avaliação clínica",
]

_embeddings_cache = None

def deve_pesquisar_web(query: str, threshold: float | None = None) -> bool:
    """
    Decide semanticamente se a query precisa de busca externa.

    - Se `threshold` não for fornecido, usa `settings.router_threshold`.
    - Retorna True quando a similaridade máxima com os exemplos for >= threshold.
    """
    global _embeddings_cache
    try:
        model = get_embeddings()
        if _embeddings_cache is None:
            docs_emb = model.embed_documents(EXEMPLOS_BUSCA)
            _embeddings_cache = np.array(docs_emb)

        query_vec = model.embed_query(query)
        query_vec = np.array(query_vec)

        scores = cosine_similarity([query_vec], _embeddings_cache)[0]
        score_max = np.max(scores)

        threshold_resolved = _get_router_threshold(threshold)
        delta = abs(score_max - threshold_resolved)
        if delta <= SCORE_ALERT_BAND:
            logger.info(f"--- 🧠 ROUTER: Score {score_max:.2f} (Limiar: {threshold_resolved}) - score near threshold (Δ={delta:.3f}) ---")
        else:
            logger.debug(f"--- 🧠 ROUTER: Score {score_max:.2f} (Limiar: {threshold_resolved}) ---")

        logger.debug("router decision: score=%.3f threshold=%.3f score_max>=threshold=%s", score_max, threshold_resolved, score_max >= threshold_resolved)
        return score_max >= threshold_resolved
    except Exception as e:
        logger.exception("Error in semantic routing")
        return False

def web_search(query: str, max_results=10) -> str:
    try:
        raw_results = []
        logger.info("🔎 Search in web: %s", query)
        with DDGS(verify=False) as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results, backend="brave"))
        if not raw_results:
            return ""
        resultados_filtrados = []
        for res in raw_results:
            link = res.get('href', '').lower()
            if any(bad in link for bad in DOMINIOS_BLOQUEADOS):
                continue
            eh_confiavel = any(good in link for good in DOMINIOS_CONFIAVEIS)
            if eh_confiavel:
                resultados_filtrados.append(res)

            if len(resultados_filtrados) >= max_results:
                break

        if not resultados_filtrados and raw_results:
            for res in raw_results:
                if not any(bad in res.get('href', '') for bad in DOMINIOS_BLOQUEADOS):
                    resultados_filtrados.append(res)
                if len(resultados_filtrados) >= 2: break
        
        contexto = []
        for res in resultados_filtrados:
            contexto.append(
                f"- Fonte: {res.get('title')}\n  URL: {res.get('href')}\n  Resumo: {res.get('body')}"
            )
        return "\n\n".join(contexto)
    except Exception:
        logger.exception("Error in web search")
        return ""