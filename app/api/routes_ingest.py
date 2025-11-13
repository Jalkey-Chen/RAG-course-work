"""
routes_ingest.py
----------------
FastAPI router for ingestion.
"""

from typing import Optional
from fastapi import APIRouter
from app.schemas.ingest import IngestRequest
from app.services.ingest_service import ingest_documents

router = APIRouter()


@router.post("/ingest", tags=["Ingest"])
def ingest(req: Optional[IngestRequest] = None):
    """
    Ingest documents from provided paths or from default directory.

    Behavior
    --------
    - If body is omitted or `paths` is empty, it will recursively
      scan `RAG_DATA_DIR` (default: 'data/raw') for supported files.
    """
    paths = req.paths if req and req.paths else None
    result = ingest_documents(paths)
    return result
