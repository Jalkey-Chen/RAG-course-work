"""
schemas/query.py
----------------
Pydantic request/response models for /query route.
"""

from pydantic import BaseModel
from typing import List, Dict


class QueryRequest(BaseModel):
    question: str
    k: int = 3


class QuerySource(BaseModel):
    rank: int
    score: float
    metadata: Dict
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[QuerySource]
