import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.utils.rag_pipeline import reindex_local_pdfs

DEFAULT_PDF_DIR = Path(os.getenv("PDF_DATA_DIR", "/data/pdfs"))


class ReindexedFile(BaseModel):
    filename: str
    chunks_indexed: int
    metadata: dict = Field(default_factory=dict)


class ReindexResponse(BaseModel):
    directory: str
    total_files: int
    total_chunks: int
    files: List[ReindexedFile]


router = APIRouter()


@router.post("", response_model=ReindexResponse)
async def reindex_endpoint(
    directory: Optional[str] = Query(
        default=None,
        description="Optional directory path containing PDFs. Defaults to /data/pdfs.",
    )
) -> ReindexResponse:
    target_dir = Path(directory) if directory else DEFAULT_PDF_DIR

    try:
        result = await reindex_local_pdfs(target_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ReindexResponse(
        directory=result["directory"],
        total_files=result["total_files"],
        total_chunks=result["total_chunks"],
        files=[ReindexedFile(**file_info) for file_info in result["files"]],
    )
