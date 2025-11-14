# BookMovieChat

BookMovieChat is a full-stack application that lets readers explore book and movie PDFs (think *Harry Potter* or *The Lord of the Rings*) through a conversational AI. Upload PDFs to the backend, generate embeddings with OpenAI + Pinecone (or an automatic local fallback), and chat via a Tailwind-styled React interface that keeps recent history and offers follow-up suggestions.

## Architecture
- **Backend**: FastAPI, LangChain, Pinecone (primary) or Chroma (local fallback), OpenAI GPT models, PyMuPDF extraction, persistent JSON history.
- **Dynamic RAG**: Query classifier adjusts top-k retrieval (specific vs broad), neighbour chunk expansion, and map–reduce summarisation for coherent, citeable answers.
- **Memory**: LangChain `ConversationSummaryBufferMemory` ensures follow-up awareness, persisted in `backend/db/history.json`.
- **Frontend**: React (Vite), Tailwind CSS dark theme, fetch-based API client with loader, suggestions, and history cards.
- **Data Flow**: PDFs → `/upload` → text extraction & chunking → embeddings stored in vector DB → `/chat` combines retrieval + GPT generation → responses cached in history.

## Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- OpenAI account + API key
- Pinecone account + API key (optional; local Chroma store activates when missing)

## Backend Setup
- **Install dependencies**
  - `cd backend`
  - `python3.11 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- **Configure environment**
  - Copy `.env.example` to `.env`
  - Fill in `OPENAI_API_KEY`, `PINECONE_*`, and optional overrides:
    - `OPENAI_CORRECTION_MODEL`, `OPENAI_SUMMARY_MODEL` for low-cost rewrites & summarisation
    - `MEMORY_TOKEN_LIMIT` to control summary buffer size
    - `PDF_DATA_DIR` directory scanned by `/reindex`
  - Set `PINECONE_API_KEY`/`PINECONE_ENV` only if you want to use Pinecone; otherwise the app persists vectors locally with Chroma
- **Run API**
  - `uvicorn backend.main:app --reload`
  - Service listens on `http://localhost:8000` with CORS enabled for all origins

### Key Endpoints
- `POST /upload` (multipart pdf): extracts with PyMuPDF, splits into 500 token chunks (50 overlap), embeds with `text-embedding-3-large`, stores vectors with metadata (file, page, neighbour chunk references).
- `POST /chat`: `{ "query": "...", "session_id": "optional" }`
  - Auto-corrects typos (`OPENAI_CORRECTION_MODEL`)
  - Classifies query (broad vs specific) → sets `k` to 25 or 5 respectively
  - Expands context with neighbouring chunks and performs map–reduce summarisation before responding
  - Returns `{ "response": "...", "sources": [{ "source", "page", "chunk_id" }], "corrected_query" }`
- `GET /history?session_id=...`: returns stored conversation for a session (last window persisted plus running summary memory).
- `GET /suggest`: returns GPT-powered prompts (falls back to a curated list when GPT unavailable).
- `POST /reindex`: re-ingests PDFs from `PDF_DATA_DIR` (defaults to `/data/pdfs`) and rebuilds embeddings.

### Document Ingestion Example
```
curl -F "file=@/path/to/book.pdf" http://localhost:8000/upload/
```

## Frontend Setup
- **Install & configure**
  - `cd frontend`
  - `npm install`
  - (optional) create `.env` with `VITE_API_BASE_URL=http://localhost:8000`
- **API expectations**
  - Chat: `POST /chat` with `{ query, session_id }`
  - History: `GET /history?session_id=...`
  - Suggestions: `GET /suggest?session_id=...`
- **Run dev server**
  - `npm run dev`
  - Visit the URL shown (default `http://localhost:5173`)
- **UI Features**
  - Dual-pane layout: chat stream + recent history cards (last 10 messages)
  - Loader animation while waiting for responses
  - Suggestion buttons (e.g., “Summarize all books”, “Who killed Snape?”) to spark the conversation
  - Automatic recall of session history via `localStorage` user id

## Storage & Persistence
- Chat history stored in `backend/db/history.json` (auto-created on start, trimmed to last 10 messages per user).
- Vector data stored in Pinecone index `book-movie-chat`; when Pinecone is unavailable, documents persist under `backend/db/chroma_store`.

## Project Structure
- `backend/`: FastAPI app, routes, utils, and persistence
- `frontend/`: Vite React client with Tailwind styling
- `backend/.env.example`: environment configuration template
- `backend/db/history.json`: default history store

## Next Steps
- Add LangGraph orchestration for more advanced dialog state handling.
- Add a `/books` endpoint to list ingested sources.
- Support multi-user authentication and scoped histories.
- Extend OCR to store thumbnails for image-heavy pages.