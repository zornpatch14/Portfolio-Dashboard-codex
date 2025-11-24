from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..schemas import FileMetadata, SelectionMeta
from ..services.stub_store import store

router = APIRouter(prefix="/api/v1", tags=["files"])


@router.get("/files", response_model=list[FileMetadata])
async def list_files() -> list[FileMetadata]:
    return store.list_files()


@router.get("/files/{file_id}", response_model=FileMetadata)
async def fetch_file(file_id: str) -> FileMetadata:
    try:
        return store.get_file(file_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")


@router.get("/selection/meta", response_model=SelectionMeta)
async def selection_meta() -> SelectionMeta:
    return store.selection_meta()
