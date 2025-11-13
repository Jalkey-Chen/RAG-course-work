"""FastAPI application entrypoint with dotenv loading."""
from dotenv import load_dotenv  # Load environment variables from .env early

# Load .env before importing modules that read environment variables
load_dotenv()

from fastapi import FastAPI
from app.api import routes_health, routes_query, routes_ingest

def create_app() -> FastAPI:
    app = FastAPI(title="RAG-course-work", version="0.1.0")

    app.include_router(routes_health.router, prefix="/api")
    app.include_router(routes_query.router, prefix="/api")
    app.include_router(routes_ingest.router, prefix="/api")

    return app


app = create_app()
