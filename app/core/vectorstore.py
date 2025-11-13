"""
vectorstore.py
---------------
Implements a minimal FAISS-based vector store for RAG.
Stores both vectors and payloads (text + metadata) and supports search.
"""

import faiss
import numpy as np
from typing import List, Dict
from pathlib import Path
import pickle


class VectorStore:
    """
    Minimal FAISS vector store that supports add, save, and search.

    Attributes
    ----------
    dim : int
        Embedding dimension (1536 for OpenAI text-embedding-3 models).
    index_path : Path
        Path to the FAISS index file.
    index : faiss.Index
        The underlying FAISS index.
    payloads : List[Dict]
        A list of dicts with shape {"text": str, "metadata": dict}.
    """

    def __init__(self, dim: int = 1536, index_path: str = "storage/faiss/index.faiss"):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim

        # Load if exists, else create fresh
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            payload_path = self.index_path.with_suffix(".payload.pkl")
            if payload_path.exists():
                with open(payload_path, "rb") as f:
                    self.payloads = pickle.load(f)
            else:
                # Backward-compat: start empty payloads if not found
                self.payloads = []
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.payloads: List[Dict] = []

    def add(self, vectors: np.ndarray, payloads: List[Dict]) -> None:
        """
        Add new vectors and their payloads.

        Parameters
        ----------
        vectors : np.ndarray
            2D array of shape (n, dim).
        payloads : List[Dict]
            Each element is {"text": str, "metadata": dict}.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors shape (n, {self.dim}), got {vectors.shape}")
        if len(payloads) != vectors.shape[0]:
            raise ValueError("vectors and payloads must have the same length")

        self.index.add(vectors)
        self.payloads.extend(payloads)

    def save(self) -> None:
        """Persist FAISS index and payloads to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.index_path.with_suffix(".payload.pkl"), "wb") as f:
            pickle.dump(self.payloads, f)

    def search(self, query_vec: np.ndarray, k: int = 3) -> List[Dict]:
        """
        Search the index for the top-k most similar chunks.

        Parameters
        ----------
        query_vec : np.ndarray
            2D array with a single query vector, shape (1, dim).
        k : int
            Number of results to return.

        Returns
        -------
        List[Dict]
            Each element: {"rank", "score", "text", "metadata"}.
        """
        if self.index.ntotal == 0:
            return []

        D, I = self.index.search(query_vec, k)
        results: List[Dict] = []
        for r, idx in enumerate(I[0]):
            if idx < len(self.payloads):
                payload = self.payloads[idx]
                results.append({
                    "rank": r + 1,
                    "score": float(D[0][r]),
                    "text": payload.get("text", ""),
                    "metadata": payload.get("metadata", {}),
                })
        return results
