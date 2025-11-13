"""
schemas/ingest.py
-----------------
Pydantic schema for ingestion.
"""

from pydantic import BaseModel
from typing import List, Optional


class IngestRequest(BaseModel):
    """
    Request body for ingestion.

    Notes
    -----
    - If `paths` is omitted or empty, the service will fallback to
      the default directory configured by env var `RAG_DATA_DIR`
      (default: 'data/raw').
    """
    paths: Optional[List[str]] = None
