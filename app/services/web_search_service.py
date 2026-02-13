"""Serviços de busca web e roteamento semântico para consultas médicas.

Este módulo fornece funcionalidades para:
- Decidir semanticamente quando realizar busca web via embeddings
- Executar buscas no DuckDuckGo com filtros de domínios confiáveis
- Formatar resultados priorizando fontes médicas verificadas

O roteamento semântico usa cosine similarity entre a query do usuário e
exemplos médicos pré-definidos para determinar se busca externa é necessária.

Attributes:
    SCORE_ALERT_BAND: Margem de alerta para scores próximos ao threshold.
    MIN_FALLBACK_LENGTH: Tamanho mínimo de resposta para fallback.
    DOMINIOS_CONFIAVEIS: Lista de domínios médicos priorizados nas buscas.
    DOMINIOS_BLOQUEADOS: Lista de domínios excluídos dos resultados.
    EXEMPLOS_BUSCA: Exemplos de consultas que requerem busca web.

Example:
    Uso típico do serviço de busca:
    
    >>> # Verificar se deve buscar
    >>> if deve_pesquisar_web("protocolos de sepse 2024"):
    ...     resultados = web_search("protocolos sepse", max_results=5)
    ...     print(resultados)
"""

import logging
import numpy as np
from ddgs import DDGS
from sklearn.metrics.pairwise import cosine_similarity
from app.services.rag_service import get_embeddings
from app.core.config import settings

logger = logging.getLogger(__name__)

SCORE_ALERT_BAND = settings.score_alert_band
"""float: Margem de alerta para scores próximos ao threshold de roteamento."""

MIN_FALLBACK_LENGTH = settings.min_fallback_length
"""int: Comprimento mínimo de resposta antes de acionar fallback."""

DOMINIOS_CONFIAVEIS = [
    ".gov.br", ".org.br", ".edu.br", "scielo.br", "pubmed.ncbi",
    "who.int", "paho.org", "msdmanuals.com", "einstein.br",
    "siriolibanes.br", "fleury.com.br", "pebmed.com.br", "medscape.com"
]
"""list[str]: Domínios médicos confiáveis priorizados nos resultados de busca."""

DOMINIOS_BLOQUEADOS = [
    "facebook.com", "instagram.com", "twitter.com", "tiktok.com",
    "youtube.com", "reddit.com", "quora.com", "yahoo.com",
    "reclameaqui.com.br", "mercadolivre.com.br", "shopee.com.br",
    "wikipedia.org"
]
"""list[str]: Domínios excluídos dos resultados por não serem fontes médicas confiáveis."""

EXEMPLOS_BUSCA = [
    "protocolos e diretrizes clínicas atualizados",
    "manejo de emergência e suporte avançado",
    "diretrizes para manejo de sepse e AVC",
    "manejo de cetoacidose e emergências metabólicas",
    "diagnóstico diferencial de sintomas comuns",
    "interações medicamentosas e ajustes de dose",
    "escores de risco e critérios de triagem"
]
"""list[str]: Exemplos de consultas médicas que tipicamente requerem busca web."""

_embeddings_cache = None

def deve_pesquisar_web(query: str, threshold: float | None = None) -> bool:
    """Decide se uma consulta deve acionar busca web via similaridade semântica.
    
    Calcula embeddings da query e compara com exemplos médicos usando cosine
    similarity. Se a similaridade máxima exceder o threshold, retorna True
    indicando que busca web é recomendada.
    
    Os embeddings dos exemplos são cacheados globalmente após o primeiro cálculo
    para otimizar performance em consultas subsequentes.

    Args:
        query: Texto da pergunta do usuário para análise.
        threshold: Limiar de decisão para busca web. Se None, usa
            `settings.router_threshold` como padrão.

    Returns:
        True se a similaridade máxima for maior ou igual ao threshold,
        indicando que busca web deve ser realizada. False em caso de erro
        durante processamento ou se a similaridade estiver abaixo do limiar.

    Note:
        O cache de embeddings (_embeddings_cache) é mantido em memória para
        toda a vida útil do processo. Em caso de erro, retorna False por
        segurança (fail-safe).

    Example:
        >>> if deve_pesquisar_web("protocolos de sepse atualizados"):
        ...     print("Realizando busca web...")
        Realizando busca web...
        
        >>> if deve_pesquisar_web("olá, como vai?"):
        ...     print("Busca não necessária")
        Busca não necessária
    """
    global _embeddings_cache
    try:
        model = get_embeddings()

        # Gera e cacheia embeddings dos exemplos na primeira execução
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
    """Executa busca no DuckDuckGo e retorna contexto filtrado de fontes confiáveis.
    
    Realiza busca web priorizando domínios médicos confiáveis e excluindo
    domínios inadequados (redes sociais, e-commerce, etc.). Os resultados são
    formatados com título, URL e resumo.
    
    Se nenhum resultado confiável for encontrado, inclui até 2 resultados não
    bloqueados como fallback para evitar respostas vazias.

    Args:
        query: Texto da consulta para busca web.
        max_results: Número máximo de resultados a considerar. Padrão é 10.

    Returns:
        String formatada contendo resultados filtrados, cada um com:
            - Fonte (título)
            - URL completa
            - Resumo do conteúdo
        
        Retorna string vazia se não houver resultados ou em caso de erro.

    Note:
        A busca usa o backend Brave do DuckDuckGo. Domínios confiáveis são
        priorizados primeiro. Se nenhum for encontrado, até 2 resultados não
        bloqueados são incluídos como fallback.
        
        Verificação SSL é desabilitada (verify=False) para evitar problemas
        com certificados em alguns ambientes.

    Example:
        >>> contexto = web_search("protocolo sepse 2024", max_results=3)
        >>> print(contexto)
        - Fonte: Protocolo de Sepse - Ministério da Saúde
          URL: https://saude.gov.br/protocolos/sepse
          Resumo: Diretrizes atualizadas para manejo de sepse...
        
        - Fonte: Sepse: Diagnóstico e Tratamento - SciELO
          URL: https://scielo.br/artigo-sepse
          Resumo: Revisão sistemática sobre diagnóstico...
    """
    try:
        logger.info("🔎 Iniciando busca web para: %s", query)
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

        # Fallback: se nenhum confiável, adiciona até 2 resultados não bloqueados
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