"""
splitter.py
------------
Split loaded documents into overlapping text chunks for embedding.
"""

from typing import List, Dict


def split_text(
    docs: List[Dict],
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Dict]:
    """
    Split loaded documents into smaller overlapping chunks.

    Parameters
    ----------
    docs : List[Dict]
        The list of raw documents from loader.py.
    chunk_size : int, optional
        Maximum number of characters per chunk.
    overlap : int, optional
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    List[Dict]
        Each chunk has:
        - 'text': str, the chunked text
        - 'metadata': dict, original metadata + chunk_id
    """
    chunks: List[Dict] = []
    for doc in docs:
        text = doc["text"]
        meta = doc["metadata"]
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {**meta, "chunk_id": chunk_id}
            })
            start += chunk_size - overlap
            chunk_id += 1

    return chunks
