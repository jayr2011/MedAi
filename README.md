# Stack Técnica - MedIA

> Assistente médica virtual baseada no modelo Gemma
> ⚠️ **Projeto para fins de teste e aprendizado**

---

## 🎯 Arquitetura

**Tipo:** Full-stack monorepo  
**Padrão:** API RESTful + SPA (Single Page Application)  
**Deploy:** Docker multi-stage build

---

## Chunking Inteligente

O MedAI usa **Semantic Chunking** para dividir documentos de forma inteligente:
- Respeita limites semânticos naturais do texto
- Evita quebrar conceitos relacionados
- Melhor recall em buscas médicas complexas

## 🐍 Backend

### Framework & Runtime Backend

- **FastAPI** `0.128.6` - Framework web assíncrono
- **Python** `3.12` - Linguagem de programação
- **Uvicorn** `0.40.0` - Servidor ASGI com suporte a HTTP/2

### Bibliotecas Core

- **Pydantic Settings** `2.10.1` - Gerenciamento de configurações
- **HTTPX** `0.28.1` - Cliente HTTP assíncrono
- **Python Multipart** `0.0.22` - Upload de arquivos
- **SlowAPI** `0.1.9` - Rate limiting

### IA & Machine Learning

- **LangChain** `1.2.9` - Framework para aplicações LLM
- **LangChain Community** `0.4.1` - Integrações da comunidade
- **LangChain Chroma** - Integração do LangChain com o Chroma
- **LangChain HuggingFace** `1.1.0` - Embeddings com modelos HF
- **ChromaDB** `1.5.0` - Vector database para embeddings
- **Sentence Transformers** `5.2.2` - Embeddings de texto
- **PyPDF** `6.7.0` - Processamento de documentos PDF
- **Llama CPP Python** - Guardrail local (Llama 3.1) para verificação de escopo
- **Hugging Face Hub** - Suporte a download de modelos
- **DDGS** `9.10.0` - Busca web (DuckDuckGo)

### Integração Externa

- **Provedor LLM OpenAI-compatible** - Hospeda o modelo Gemma via API

### Estrutura do Backend

```plaintext
app/
├── api/
│   ├── deps.py           # Injeção de dependências
│   └── v1/
│       ├── endpoints/    # Rotas da API
│       └── schemas/      # Validação de dados
├── core/
│   └── config.py         # Configurações centralizadas
└── services/
  ├── llm_service.py         # Cliente do modelo
  ├── rag_service.py         # RAG com PDFs e ChromaDB
  └── web_search_service.py  # Busca web e roteamento semântico
```

---

## 🎨 Frontend

### Framework & Runtime Frontend

- **Vue 3** `3.5.27` - Framework JavaScript progressivo
- **TypeScript** `5.9.3` - Superset tipado do JavaScript
- **Vite** `7.3.1` - Build tool e dev server
- **Node.js** `^20.19.0 || >=22.12.0`

### Gerenciamento de Estado & Roteamento

- **Vue Router** `5.0.2` - Roteamento SPA
- **Pinia** `3.0.4` - State management oficial do Vue 3

### Estilização

- **Tailwind CSS** `4.1.18` - Framework CSS utility-first
- **@tailwindcss/typography** `0.5.19` - Plugin para formatação de texto
- **@tailwindcss/vite** `4.1.18` - Integração com Vite

### Renderização de Conteúdo

- **Marked** `17.0.1` - Parser Markdown para exibição de respostas

### Qualidade de Código

- **ESLint** `9.39.2` - Linter JavaScript/TypeScript
- **Prettier** `3.8.1` - Formatador de código
- **Oxlint** `1.42.0` - Linter de alta performance
- **Vue TSC** `3.2.4` - Type checking para Vue

### DevTools

- **Vite Plugin Vue DevTools** `8.0.5` - Ferramentas de debug

### SEO & Acessibilidade

- **Meta tags** (description/canonical) e ícones
- **robots.txt** - Controle de indexação
- **ARIA e roles** em componentes de chat

### Estrutura

```plaintext
frontend/src/
├── assets/          # CSS global
├── components/      # Componentes reutilizáveis
│   ├── ChatBubble.vue
│   └── ChatInput.vue
├── composables/     # Lógica reutilizável (Composition API)
│   ├── useChat.ts
│   └── useDark.ts
├── routes/          # Configuração de rotas
├── stores/          # Estado global (Pinia)
├── types/           # Definições TypeScript
└── views/           # Páginas/telas
    └── ChatView.vue
```

---

## 🐳 DevOps

### Containerização

- **Docker** - Multi-stage build
  - Stage 1: Build do frontend (Node 20 Alpine)
  - Stage 2: Runtime Python + frontend estático

### CI/CD

- **GitHub Actions** (potencial, estrutura pronta)

---

## 🔧 Configurações

### Backend

- **CORS:** Habilitado para todas as origens
- **Streaming:** Suporte a SSE (Server-Sent Events)
- **SSL:** Verificação configurável para provedor LLM (dev/prod)
- **Rate Limiting:** Implementado via SlowAPI
- **Guardrail local:** Classificação de escopo com Llama 3.1 (CPU)
- **RAG:** Ingestão e busca de documentos PDF via ChromaDB
- **Busca web:** Roteamento semântico e filtros de domínios
- **Docstrings:** Uso dos docstrings para documentação de API e métodos

### Frontend

- **Dark Mode:** Suporte nativo com persistência
- **Markdown:** Renderização de respostas do modelo
- **Auto-scroll:** UI otimizada para chat
- **Acessibilidade:** Live regions, labels e roles
- **Type Safety:** TypeScript strict mode

### Modelo LLM

- **Max Tokens:** 1024
- **Temperature:** 0.5
- **Top P:** 0.7
- **Streaming:** Habilitado

---

## 📦 Build & Deploy

### Desenvolvimento Local

**Backend:**

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

### Produção

**Build Frontend:**

```bash
cd frontend
npm run build
```

**Docker:**

```bash
docker build -t media .
docker run -p 8000:8000 --env-file .env media
```

---

## 🌐 Endpoints

- `GET /health` - Health check da aplicação
- `POST /v1/chat/stream` - Chat com streaming de resposta
- `POST /v1/rag/ingest` - Upload e ingestão de PDFs
- `GET /v1/rag/documents` - Listar documentos ingeridos
- `DELETE /v1/rag/documents/{file_name}` - Remover documento do RAG
- `GET /` - Frontend SPA (produção)

---

## 🔐 Variáveis de Ambiente

```bash
LLM_BASE_URL=<url-base-openai-compatible>
LLM_API_KEY=<token-ou-chave-de-acesso>
LLM_MODEL=<id-do-modelo-ou-rota>
HUGGINGFACE_TOKEN=<token-hf-opcional>
MAX_TOKENS=1024
ROUTER_THRESHOLD=0.5
MIN_FALLBACK_LENGTH=50
SCORE_ALERT_BAND=0.05
```

---

**Versão do Stack:** 1.1  
**Última Atualização:** Fevereiro 2026
