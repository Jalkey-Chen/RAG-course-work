"""
routes_query.py
----------------
FastAPI router for querying the RAG system.
"""

from fastapi import APIRouter
from app.schemas.query import QueryRequest, QueryResponse
from app.services.query_service import answer_question

router = APIRouter()


@router.post("/query", response_model=QueryResponse, tags=["Query"])
def query(req: QueryRequest):
    """
    Handle user question:
    - retrieve top-k contexts
    - generate concise answer with citations
    """
    result = answer_question(req.question, req.k)
    return QueryResponse(**result)
