"""
retriever.py
-------------
Thin wrapper that embeds a query and searches the VectorStore.
"""

import numpy as np
from typing import List, Dict
from app.core.embeddings import Embedder
from app.core.vectorstore import VectorStore


class Retriever:
    """
    Retriever that uses the Embedder + VectorStore to return top-k contexts.
    """

    def __init__(self, dim: int = 1536):
        self.embedder = Embedder()
        self.vs = VectorStore(dim=dim)

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """
        Embed a query and search for top-k relevant chunks.

        Parameters
        ----------
        query : str
            Natural-language question.
        k : int
            Number of contexts to retrieve.

        Returns
        -------
        List[Dict]
            Each context: {"rank", "score", "text", "metadata"}.
        """
        qvec = self.embedder.embed_texts([query]).astype(np.float32)
        return self.vs.search(qvec, k=k)
