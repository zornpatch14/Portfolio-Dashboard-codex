from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..schemas import FileUploadResponse
from ..services.data_store import store

router = APIRouter(prefix="/api/v1", tags=["upload"])


@router.post("/upload", response_model=FileUploadResponse)
async def upload_files(files: List[UploadFile] = File(...)) -> FileUploadResponse:
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")

    filenames = [file.filename for file in files]
    return store.ingest(filenames)
