"""
embeddings.py
--------------
Provides a robust wrapper around the OpenAI embedding API with
safe batching by total tokens and max items per request.
"""

from typing import List
import os
import numpy as np
from openai import OpenAI
import tiktoken

def _approx_token_count(text: str, model: str) -> int:
    """
    Approximate token count for a given text and model using tiktoken.

    Notes
    -----
    - This is an approximation but sufficient for batching safeguards.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


class Embedder:
    """
    Wrapper class for OpenAI text embedding with safe batching.

    Batching rules
    --------------
    - MAX_BATCH_TOKENS: upper bound of total tokens per API call.
    - MAX_BATCH_ITEMS: upper bound of items per API call.
    - MAX_CHUNK_TOKENS: truncate any single text longer than this.
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")

        # Conservative limits to avoid 400: max_tokens_per_request
        self.MAX_BATCH_TOKENS = int(os.getenv("RAG_EMBED_MAX_BATCH_TOKENS", "180000"))
        self.MAX_BATCH_ITEMS = int(os.getenv("RAG_EMBED_MAX_BATCH_ITEMS", "96"))
        # Truncate overly long chunks defensively (per-item)
        self.MAX_CHUNK_TOKENS = int(os.getenv("RAG_EMBED_MAX_CHUNK_TOKENS", "8000"))

    def _truncate(self, text: str) -> str:
        """Truncate a single text to MAX_CHUNK_TOKENS if necessary."""
        tokens = _approx_token_count(text, self.model)
        if tokens <= self.MAX_CHUNK_TOKENS:
            return text
        # Cheap truncation by characters (good enough for MVP).
        # For stricter control you can iteratively trim by tokens.
        ratio = self.MAX_CHUNK_TOKENS / max(tokens, 1)
        cut = max(200, int(len(text) * ratio))
        return text[:cut]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of text strings into embedding vectors with safe batching.

        Parameters
        ----------
        texts : List[str]
            Input text strings.

        Returns
        -------
        np.ndarray
            A 2D array of embedding vectors (float32).
        """
        # 1. Preprocess: drop empty strings and truncate long ones
        prepped: List[str] = []
        for t in texts:
            if not t or not t.strip():
                continue
            prepped.append(self._truncate(t.strip()))

        if not prepped:
            return np.zeros((0, 1536), dtype=np.float32)

        # 2. Batch by (a) total tokens, (b) max items
        batches: List[List[str]] = []
        cur_batch: List[str] = []
        cur_tokens = 0

        for t in prepped:
            t_tokens = _approx_token_count(t, self.model)

            # If adding this item exceeds either constraint, flush batch
            if (
                cur_batch
                and (
                    cur_tokens + t_tokens > self.MAX_BATCH_TOKENS
                    or len(cur_batch) >= self.MAX_BATCH_ITEMS
                )
            ):
                batches.append(cur_batch)
                cur_batch = []
                cur_tokens = 0

            cur_batch.append(t)
            cur_tokens += t_tokens

        if cur_batch:
            batches.append(cur_batch)

        # 3. Call API per batch and collect vectors
        all_vecs: List[np.ndarray] = []
        for batch in batches:
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            all_vecs.append(vecs)

        return np.vstack(all_vecs)
