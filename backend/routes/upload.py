from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.utils.rag_pipeline import ingest_document


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    metadata: dict


router = APIRouter()


@router.post("", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type not in {"application/pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        result = await ingest_document(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Failed to ingest PDF.") from exc

    return UploadResponse(**result)

