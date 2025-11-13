"""
routes_web.py
--------------
Serve minimal HTML UI for the RAG app using Jinja templates.
"""

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.services.query_service import answer_question

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Render home page with a text box for query input."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...)):
    """Handle form submission and show the answer page."""
    result = answer_question(question, k=3)
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "question": question, **result},
    )
