"""
ingest_service.py
-----------------
Integrates document expansion, loading, splitting, embedding, and storing.
"""

from app.core.loader import load_documents, expand_input_paths
from app.core.splitter import split_text
from app.core.embeddings import Embedder
from app.core.vectorstore import VectorStore


def ingest_documents(paths: list[str] | None) -> dict:
    """
    Full ingestion pipeline: expand -> load -> split -> embed -> store.

    Notes
    -----
    - If `paths` is None or empty, we will expand from `RAG_DATA_DIR`
      recursively and include all supported file types.
    """
    file_list = expand_input_paths(paths)
    docs = load_documents(file_list)
    chunks = split_text(docs)

    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_texts(texts)

    vs = VectorStore()
    payloads = [{"text": c["text"], "metadata": c["metadata"]} for c in chunks]
    vs.add(vectors, payloads)
    vs.save()

    return {
        "expanded_files": len(file_list),
        "loaded_docs": len(docs),
        "total_chunks": len(chunks),
    }
