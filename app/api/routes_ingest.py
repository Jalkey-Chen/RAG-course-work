from fastapi import APIRouter
from app.schemas.ingest import IngestRequest
from app.services.ingest_service import ingest_documents

router = APIRouter()

@router.post("/ingest", tags=["Ingest"])
def ingest(req: IngestRequest):
    result = ingest_documents(req.paths)
    return result
