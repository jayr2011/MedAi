# Stack Técnica - MedIA

> Assistente médica virtual baseada no modelo Gemma
> ⚠️ **Projeto para fins de teste e aprendizado**

---

## 🎯 Arquitetura

**Tipo:** Full-stack monorepo  
**Padrão:** API RESTful + SPA (Single Page Application)  
**Deploy:** Docker multi-stage build

---

## 🐍 Backend

### Framework & Runtime Backend

- **FastAPI** `0.115.0` - Framework web assíncrono
- **Python** `3.12` - Linguagem de programação
- **Uvicorn** `0.30.6` - Servidor ASGI com suporte a HTTP/2

### Bibliotecas Core

- **Pydantic Settings** `2.5.2` - Gerenciamento de configurações
- **HTTPX** `0.27.0` - Cliente HTTP assíncrono
- **Python Multipart** `0.0.9` - Upload de arquivos
- **SlowAPI** `0.1.9` - Rate limiting

### IA & Machine Learning

- **LangChain** - Framework para aplicações LLM
- **LangChain Community** - Integrações da comunidade
- **ChromaDB** - Vector database para embeddings
- **Sentence Transformers** - Embeddings de texto
- **PyPDF** - Processamento de documentos PDF

### Integração Externa

- **Databricks** - Hospeda o modelo Gemma via API

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
    └── databricks_service.py  # Cliente do modelo
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
- **SSL:** Verificação desabilitada para Databricks (dev)
- **Rate Limiting:** Implementado via SlowAPI

### Frontend

- **Dark Mode:** Suporte nativo com persistência
- **Markdown:** Renderização de respostas do modelo
- **Auto-scroll:** UI otimizada para chat
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
- `GET /` - Frontend SPA (produção)

---

## 🔐 Variáveis de Ambiente

```bash
DATABRICKS_URL=<url-do-endpoint>
DATABRICKS_TOKEN=<token-de-acesso>
MAX_TOKENS=1024
```

---

**Versão do Stack:** 1.0  
**Última Atualização:** Fevereiro 2026
