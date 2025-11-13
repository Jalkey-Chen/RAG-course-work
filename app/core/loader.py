"""
loader.py
----------
Provides utilities to load text content from multiple file types:
txt, md, docx, csv, pdf, and html. Also supports expanding directories.
"""

from pathlib import Path
from typing import List, Dict
import csv
import re
import os

from bs4 import BeautifulSoup  # For HTML parsing
from PyPDF2 import PdfReader   # For PDF text extraction

try:
    from docx import Document  # Optional dependency for .docx
except ImportError:
    Document = None

# Supported extensions
SUPPORTED_EXTS = {".txt", ".md", ".docx", ".csv", ".pdf", ".html"}


def clean_text(text: str) -> str:
    """
    Clean extra whitespace and control characters.

    Returns
    -------
    str
        Cleaned text with normalized whitespace.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def expand_input_paths(paths: List[str] | None) -> List[str]:
    """
    Expand a list of inputs (files/dirs/globs) into concrete file paths.

    Parameters
    ----------
    paths : List[str] | None
        If None or empty, will use env var `RAG_DATA_DIR` (default: 'data/raw').

    Returns
    -------
    List[str]
        Concrete file paths filtered by SUPPORTED_EXTS, de-duplicated.
    """
    # Fallback to default directory when not provided
    if not paths:
        base_dir = os.getenv("RAG_DATA_DIR", "data/raw")
        paths = [base_dir]

    out: list[str] = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTS:
                out.append(str(path))
        elif path.is_dir():
            for f in path.rglob("*"):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                    out.append(str(f))
        else:
            # Treat as glob pattern (e.g., "data/raw/**/*.pdf")
            for f in Path().glob(p):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                    out.append(str(f))

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for p in out:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return deduped


def load_documents(paths: List[str]) -> List[Dict]:
    """
    Load text content from multiple file types.

    Parameters
    ----------
    paths : List[str]
        A list of file paths to load (should be expanded beforehand).

    Returns
    -------
    List[Dict]
        Each element has keys:
        - 'text': str, the extracted text
        - 'metadata': dict, including file name and type
    """
    docs: List[Dict] = []

    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] File not found: {p}")
            continue

        ext = path.suffix.lower()
        text = ""

        # --- TXT / Markdown ---
        if ext in [".txt", ".md"]:
            text = path.read_text(encoding="utf-8", errors="ignore")

        # --- DOCX ---
        elif ext == ".docx" and Document is not None:
            doc = Document(path)
            text = "\n".join([para.text for para in doc.paragraphs])

        # --- CSV ---
        elif ext == ".csv":
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = [" | ".join(row) for row in reader]
                text = "\n".join(rows)

        # --- PDF ---
        elif ext == ".pdf":
            pdf = PdfReader(path)
            pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n".join(pages)

        # --- HTML ---
        elif ext == ".html":
            html = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=" ")

        else:
            print(f"[WARN] Unsupported file type: {ext}")
            continue

        text = clean_text(text)
        if text:
            docs.append({
                "text": text,
                "metadata": {"source": str(path), "type": ext}
            })

    return docs
