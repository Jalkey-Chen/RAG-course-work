"""
query_service.py
----------------
Wire retriever + generator for end-to-end RAG answering.
"""

from typing import Dict, List
from app.core.retriever import Retriever
from app.core.generator import Generator


def answer_question(question: str, k: int = 3) -> Dict:
    """
    Retrieve top-k contexts and generate an answer.

    Returns
    -------
    dict
        {
          "answer": str,
          "sources": List[{"rank","score","metadata","text"}]
        }
    """
    retriever = Retriever()
    contexts = retriever.retrieve(question, k=k)

    generator = Generator()
    response = generator.answer(question, contexts)

    # Optionally shrink raw text in sources for response payload
    compact_sources: List[Dict] = []
    for c in contexts:
        shortened = dict(c)
        shortened["text"] = c.get("text", "")[:400]
        compact_sources.append(shortened)

    return {"answer": response, "sources": compact_sources}
