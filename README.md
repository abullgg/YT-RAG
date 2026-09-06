# RAG Document Assistant

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js_14-000000?style=flat-square&logo=next.js&logoColor=white)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)

A **local-first**, privacy-preserving document Q&A system. Upload a PDF or text file, ask questions in plain English, and receive answers grounded strictly in your document — with citations pointing to the exact section they came from.

No cloud. No API keys. Nothing leaves your machine.

---

## Overview

Standard RAG (Retrieval-Augmented Generation) systems struggle with two core problems: **fragmented chunking** (breaking tables or splitting mid-sentence) and **shallow retrieval** (pure vector search misses exact terms, codes, and figures).

This project solves both through a purpose-built ingestion pipeline and a two-stage retrieval engine.

---

## How It Works

### Document Ingestion

When a file is uploaded, it passes through a **4-pass chunking pipeline** before a single embedding is created:

| Pass | Name | What It Does |
|------|------|--------------|
| 0 | **Atomic Isolation** | Tables and code blocks are extracted and shielded from splitting. |
| 1 | **Header Normalization** | A confidence-scored detector identifies section boundaries and preserves document hierarchy. |
| 2 | **Semantic Splitting** | Content is divided using Markdown-aware + recursive strategies, keeping related sentences together. |
| 3 | **Metadata Enrichment** | Every chunk is tagged with its H1 → H2 → H3 breadcrumb, block type, and character position. |

The resulting chunks are embedded using `BAAI/bge-base-en-v1.5` and stored in a FAISS index alongside a BM25 keyword index.

---

### Query & Retrieval

When a question is submitted, retrieval runs in two stages:

**Stage 1 — Hybrid Fusion Search**
The query is matched simultaneously against:
- A **FAISS dense index** (semantic similarity via BGE embeddings)
- A **BM25 sparse index** (exact keyword matching)

Results are fused at a 70/30 ratio, ensuring that both conceptual matches *and* precise terms are captured.

**Stage 2 — Cross-Encoder Reranking**
The top candidates from Stage 1 are re-scored by `BAAI/bge-reranker-base`, a Cross-Encoder that reads the query and each passage *together* — providing significantly higher relevance precision than a bi-encoder alone.

**Context Budgeting**
The final shortlist is trimmed to a configurable character limit before being sent to the LLM, ensuring the model always receives focused, high-signal context.

---

### Answer Generation

The selected passages are sent to a locally-running **Qwen3 4B** model via Ollama with a strict grounding prompt. The model is instructed to answer only from the provided context and to clearly state when information is not available.

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Frontend | Next.js 14 (App Router) |
| LLM | Ollama — `qwen3:4b` |
| Embeddings | `BAAI/bge-base-en-v1.5` (768-dim) |
| Vector Store | FAISS `IndexFlatIP` |
| Keyword Search | BM25 (`rank-bm25`) |
| Reranker | `BAAI/bge-reranker-base` |
| PDF Parsing | pdfplumber |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com/) installed and running

### 1. Clone & Install

```bash
git clone https://github.com/abullgg/RAG_Doc_Assist.git
cd RAG_Doc_Assist

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pull the Model

```bash
ollama pull qwen3:4b
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=qwen3:4b
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768
MAX_CONTEXT_CHARS=12000
```

> **Qwen3 optimizations applied automatically**
> - `think=False` is passed on every Ollama request to suppress chain-of-thought
>   reasoning (saves tokens; not useful for grounded RAG).
> - Context is wrapped in `<context>` XML tags (Qwen3 attends to these more reliably
>   than plain ASCII fences).
> - `MAX_CONTEXT_CHARS` is set to 12,000 (was 3,000 for Gemma's 8K window);
>   Qwen3 4B has a 32K token window so this is still very conservative.

### 4. Run

```bash
# Backend
uvicorn src.main:app --reload

# Frontend (in a separate terminal)
cd frontend
npm install
npm run dev
```

- **API**: `http://localhost:8000` — Interactive docs at `/docs`
- **UI**: `http://localhost:3000`

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload a PDF or TXT file for indexing |
| `POST` | `/ask` | Ask a question against an indexed document |
| `GET` | `/health` | System health and indexed document count |
| `GET` | `/status/{job_id}` | Poll the progress of a background upload job |

**`POST /ask` — Request body:**
```json
{
  "question": "string",
  "document_id": "optional-uuid",
  "top_k": 3,
  "max_context_chars": 3000
}
```

**Response includes:** `answer`, `source_chunks` (with breadcrumbs and confidence scores), `context_chars_used`.

---

## Notes

- **Switching embedding models** requires deleting `./data/` and re-uploading documents. The persistence layer detects dimension mismatches at startup automatically.
- **All data is stored locally** in `./data/` — FAISS index, chunk metadata, and document registry.
- On first startup, the BGE embedding model (~420 MB) and reranker (~280 MB) are downloaded automatically from HuggingFace.

---

## License

MIT — use it however you like.
