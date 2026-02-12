"""Serviços de busca web e roteamento semântico.

Fornece duas responsabilidades principais:
- decidir semanticamente quando realizar busca web (`deve_pesquisar_web`);
- executar buscas no DuckDuckGo com filtros e formatação (`web_search`).

O módulo utiliza embeddings para comparar a query com exemplos médicos e
DDGS para obter resultados do mecanismo de busca.
"""

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
    """Decide se uma consulta deve acionar busca web externa.

    A decisão é baseada em similaridade semântica entre a `query` e um conjunto
    de exemplos médicos (`EXEMPLOS_BUSCA`). Usa embeddings para calcular a
    similaridade e compara a similaridade máxima com `threshold`.

    Args:
        query (str): texto da pergunta do usuário.
        threshold (float | None): limiar opcional para tomada de decisão. Se
            None, `settings.router_threshold` é utilizado.

    Returns:
        bool: True quando a similaridade máxima >= limiar; False em caso de
            erro ou quando abaixo do limiar.
    """
    global _embeddings_cache
    try:
        # Recupera/gera embeddings dos exemplos (cache in-process)
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
    """Executa busca no DuckDuckGo e retorna contexto filtrado e formatado.

    A função prioriza domínios confiáveis médicos e exclui domínios listados em
    `DOMINIOS_BLOQUEADOS`. Se nenhum resultado confiável for encontrado, uma
    pequena amostra de resultados não bloqueados pode ser retornada.

    Args:
        query (str): texto da pesquisa.
        max_results (int): número máximo de resultados a considerar.

    Returns:
        str: contexto formatado com título, URL e resumo; string vazia em caso
            de erro ou ausência de resultados.
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
            
            # Exclui domínios bloqueados imediatamente
            if any(bad in link for bad in DOMINIOS_BLOQUEADOS):
                continue
            
            # Prioriza domínios confiáveis
            if any(good in link for good in DOMINIOS_CONFIAVEIS):
                resultados_filtrados.append(res)

            if len(resultados_filtrados) >= max_results:
                break

        # Se nenhum resultado confiável, relaxa filtro e adiciona primeiras
        # páginas não bloqueadas (fallback reduzido)
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