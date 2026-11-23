# YT-RAG: YouTube Video & Document Intelligence Assistant

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js_16-000000?style=flat-square&logo=next.js&logoColor=white)](https://nextjs.org/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-white?style=flat-square&logo=ollama&logoColor=black)](https://ollama.com/)
[![FAISS](https://img.shields.io/badge/Vector_Store-FAISS-00599C?style=flat-square)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)

A **local-first, privacy-preserving multimodal RAG system** designed for both **YouTube video transcripts** and **documents (PDF/TXT)**. Ingest any YouTube video or document, ask questions in natural language, and receive grounded answers with **exact deep-linked video timestamps** (e.g. `https://youtube.com/watch?v=...&t=145s`) and document section breadcrumbs.

**100% Local & Private:** No cloud APIs, no external subscriptions, and no data leaving your machine.

---

## Key Highlights

- **Speech-Boundary YouTube Chunker**: Ingests creator or auto-generated YouTube captions, segments them into ~45-second semantic windows snapped to natural speech pauses, and preserves millisecond-accurate timestamps.
- **Deep-Linked Video Citations**: Every retrieved answer citation includes a clickable YouTube timestamp link (`[MM:SS - MM:SS]`) that opens the exact second in the video player.
- **4-Pass Document Ingestion Pipeline**: Ingests PDFs and TXT files while shielding tables, preserving Markdown/heading hierarchy ($H_1 \to H_2 \to H_3$), and tagging chunk breadcrumbs.
- **Two-Stage Hybrid Retrieval & Reranking**:
  1. **Hybrid Fusion (70/30)**: Fuses dense semantic search (`BAAI/bge-base-en-v1.5` over FAISS) with sparse lexical search (BM25) for high recall on both concepts and exact terms.
  2. **Cross-Encoder Reranking**: Re-evaluates top candidates using `BAAI/bge-reranker-base` for ultra-precise relevance scoring.
- **Fast Local CPU Inference**: Powered by Ollama with `qwen2.5:1.5b` (with token-capping for ~2–4s responses) or `qwen3:4b`, with strict grounding prompts to eliminate hallucinations.
- **Instant Persistence & Idempotency**: FAISS indices, metadata, and chunk stores are persisted in `./data/`. Videos and documents are indexed once and cached permanently.
- **Interactive Modern UI**: Dual-tab Next.js 16 (Turbopack) frontend with real-time indexing status, streaming-friendly display, and timestamp jump links.

---

## Architecture Overview

```
                      ┌─────────────────────────────────────────┐
                      │             User Interface              │
                      │        (Next.js 16 + Turbopack)         │
                      └────────────────────┬────────────────────┘
                                           │
                        REST API (FastAPI @ Port 8000)
                                           │
      ┌────────────────────────────────────┴────────────────────────────────────┐
      ▼                                                                         ▼
[ YouTube Ingestion ]                                                 [ Document Ingestion ]
  • youtube-transcript-api                                              • pdfplumber / txt
  • Creator & Auto-captions                                             • Pass 0: Table & Code Isolation
  • Speech-Boundary Chunker                                             • Pass 1: Header Normalization
    (~45s window, ~10s overlap)                                         • Pass 2: Semantic Splitting
  • Canonical Timestamps & URLs                                         • Pass 3: Metadata Enrichment
      │                                                                         │
      └────────────────────────────────────┬────────────────────────────────────┘
                                           │
                              ┌────────────┴────────────┐
                              │  BGE Bi-Encoder (768d)  │
                              └────────────┬────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
          [ FAISS Dense Index ]                       [ BM25 Sparse Index ]
          Cosine Similarity (IP)                       Keyword Matching
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           │
                                ┌──────────┴──────────┐
                                │ Hybrid Fusion (70/30)│
                                └──────────┬──────────┘
                                           │ Top Candidates
                                           ▼
                             ┌───────────────────────────┐
                             │ Cross-Encoder Reranker    │
                             │ (BAAI/bge-reranker-base)  │
                             └─────────────┬─────────────┘
                                           │ Top-K Relevance Shortlist
                                           ▼
                             ┌───────────────────────────┐
                             │  Context Budget Trimmer   │
                             │   (MAX_CONTEXT_CHARS)     │
                             └─────────────┬─────────────┘
                                           │ Grounded Context
                                           ▼
                             ┌───────────────────────────┐
                             │   Ollama Local LLM        │
                             │ (qwen2.5:1.5b / qwen3:4b) │
                             └─────────────┬─────────────┘
                                           │
                                           ▼
                       Grounded Answer + Deep-Linked Citations
```

---

## Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Backend API** | FastAPI + Uvicorn | Asynchronous REST API with background task queues |
| **Frontend UI** | Next.js 16 (App Router) | Reactive UI with Turbopack, Tailwind, and React Markdown |
| **LLM Inference** | Ollama (`qwen2.5:1.5b`) | Fast local generation (~2–4s latency on CPU, 600-token cap) |
| **Alternative LLM** | Ollama (`qwen3:4b`) | Advanced local model for deeper analytical synthesis |
| **Embedding Model** | `BAAI/bge-base-en-v1.5` | 768-dimensional sentence transformer (local HuggingFace) |
| **Vector Index** | FAISS (`IndexFlatIP`) | Inner product similarity on $L_2$-normalized dense vectors |
| **Keyword Index** | BM25 (`rank-bm25`) | Lexical scoring for exact acronyms, names, and identifiers |
| **Reranker** | `BAAI/bge-reranker-base` | Cross-encoder for full-attention query-passage relevance |
| **Video Processing**| `youtube-transcript-api` | Captions extraction with speech-pause boundary chunker |
| **PDF Extraction**  | `pdfplumber` | Layout-aware text and tabular data extraction |
| **Persistence**     | Pickle + FAISS binary + JSON | Permanent local caching under `./data/` |

---

## Getting Started

### Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 18.x or higher (with `npm`)
- **[Ollama](https://ollama.com/)**: Installed and running locally

---

### 1. Clone & Set Up Python Environment

```bash
git clone https://github.com/abullgg/YT-RAG.git
cd YT-RAG

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r requirements.txt
```

---

### 2. Pull the Local LLM via Ollama

Ensure Ollama is running (`ollama serve`), then pull the recommended lightweight model:

```bash
# Recommended: Ultra-fast sub-4s CPU inference
ollama pull qwen2.5:1.5b

# Optional: Higher-capacity 4B model
ollama pull qwen3:4b
```

---

### 3. Environment Configuration

Create or update your `.env` file in the project root:

```env
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=qwen2.5:1.5b
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768
MAX_CONTEXT_CHARS=12000
```

> [!NOTE]
> - `MODEL_NAME=qwen2.5:1.5b` delivers **2–4 second response times** on CPU with a strict output token limit (`num_predict: 600`).
> - `MAX_CONTEXT_CHARS=12000` provides generous context while preventing model context exhaustion.
> - On first launch, the BGE embedding model (~420 MB) and Cross-Encoder reranker (~280 MB) download automatically from HuggingFace.

---

### 4. Run the Project

You can start the backend and frontend in separate terminals:

#### Terminal 1 — Start Ollama & Backend
```bash
# Start Ollama (if not already running in background)
ollama serve

# Start FastAPI backend
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2 — Start Frontend UI
```bash
cd frontend
npm install
npm run dev
```

- **Web Application**: Open [http://localhost:3000](http://localhost:3000)
- **FastAPI Documentation**: Open [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Check**: `http://localhost:8000/health`

---

## API Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/videos/ingest` | Ingest and index a YouTube video transcript |
| `POST` | `/upload` | Upload and index a PDF or TXT document |
| `POST` | `/ask` | Submit a grounded question against any indexed document or video |
| `GET` | `/health` | Check backend health and total indexed document count |
| `GET` | `/status/{job_id}` | Poll progress for background upload/indexing jobs |

---

### Example API Usage

#### 1. Ingest a YouTube Video
```bash
curl -X POST "http://localhost:8000/videos/ingest" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://www.youtube.com/watch?v=SVacJl1ruuY"}'
```

**Response:**
```json
{
  "document_id": "yt_SVacJl1ruuY",
  "video_id": "SVacJl1ruuY",
  "video_title": "Honda Civic, VW Polo, Innova, Fortuner: Indian Cars of 2006-2010",
  "channel": "evo India",
  "status": "completed",
  "chunks_created": 54
}
```

#### 2. Ask a Grounded Question
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What does the video say about the Skoda Yeti?",
       "document_id": "yt_SVacJl1ruuY",
       "top_k": 3
     }'
```

**Response snippet:**
```json
{
  "answer": "The Skoda Yeti is highlighted as a mid-size SUV featuring a Haldex clutch 4x4 system that gave it serious off-road capability despite its compact hatchback proportions...",
  "source_chunks": [
    {
      "source_type": "youtube",
      "video_title": "Honda Civic, VW Polo, Innova, Fortuner...",
      "source_label": "Honda Civic, VW Polo, Innova, Fortuner... [27:53 - 29:15]",
      "timestamp_url": "https://www.youtube.com/watch?v=SVacJl1ruuY&t=1673s",
      "start_seconds": 1673.6,
      "end_seconds": 1755.04,
      "relevance_score": 0.892
    }
  ],
  "context_chars_used": 2840
}
```

---

## Troubleshooting & FAQ

### YouTube Transcript Rate-Limiting (`IpBlocked`)
* **Why it happens**: YouTube restricts unauthenticated caption scrapers when multiple requests originate from the same IP (or from shared college/VPN networks).
* **Fix**:
  1. **Hotspot Switch**: Connect your laptop to a mobile phone hotspot — cellular carriers grant a fresh residential IP immediately.
  2. **Cooldown**: Wait 15–30 minutes for YouTube's temporary rate-limit window to expire.
  3. **Local Cache**: Videos indexed once into `./data/` are stored permanently; querying previously indexed videos never touches YouTube servers.

### Switching Embedding Models
* The FAISS index dimension must strictly match `EMBEDDING_DIMENSION` (e.g. 768 for `bge-base`, 1024 for `bge-large`).
* If you change `EMBEDDING_MODEL` in `.env`, delete `./data/` before starting the server so the index can re-initialize cleanly.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
