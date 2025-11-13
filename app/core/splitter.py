"""
splitter.py
------------
Split loaded documents into overlapping text chunks for embedding.
"""

from typing import List, Dict


def split_text(
    docs: List[Dict],
    chunk_size: int = 800,     # ↓ smaller to reduce tokens per chunk
    overlap: int = 120
) -> List[Dict]:
    """
    Split loaded documents into smaller overlapping chunks.

    Parameters
    ----------
    docs : List[Dict]
        The list of raw documents from loader.py.
    chunk_size : int, optional
        Max characters per chunk (smaller is safer for embedding limits).
    overlap : int, optional
        Overlapping characters between consecutive chunks.

    Returns
    -------
    List[Dict]
        Each chunk has:
        - 'text': str
        - 'metadata': dict (original metadata + chunk_id)
    """
    chunks: List[Dict] = []
    for doc in docs:
        text = doc["text"]
        meta = doc["metadata"]
        if not text:
            continue

        start = 0
        chunk_id = 0
        step = max(1, chunk_size - overlap)

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {**meta, "chunk_id": chunk_id}
                })
            start += step
            chunk_id += 1

    return chunks
