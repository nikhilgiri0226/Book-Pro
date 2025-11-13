from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.chat import router as chat_router
from backend.routes.reindex import router as reindex_router
from backend.routes.upload import router as upload_router
from backend.utils.memory_manager import memory_manager
from backend.utils.rag_pipeline import init_vector_store


def create_app() -> FastAPI:
    app = FastAPI(
        title="BookMovieChat",
        description=(
            "Conversational AI for chatting about books and movies with PDF ingestion "
            "backed by GPT and Pinecone."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event() -> None:
        init_vector_store()
        memory_manager.load_history()

    app.include_router(upload_router, prefix="/upload", tags=["upload"])
    app.include_router(chat_router, tags=["chat"])
    app.include_router(reindex_router, prefix="/reindex", tags=["reindex"])

    return app


app = create_app()

