from fastapi import APIRouter
from app.schemas.query import QueryRequest, QueryResponse
from app.services.query_service import answer_question

router = APIRouter()

@router.post("/query", response_model=QueryResponse, tags=["Query"])
def query(req: QueryRequest):
    answer = answer_question(req.question, req.k)
    return QueryResponse(answer=answer)
