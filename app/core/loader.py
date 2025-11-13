"""
loader.py
----------
Provides utilities to load text content from multiple file types:
txt, md, docx, csv, pdf, and html.
"""

from pathlib import Path
from typing import List, Dict
import csv
import re

from bs4 import BeautifulSoup  # For HTML parsing
from PyPDF2 import PdfReader   # For PDF text extraction

try:
    from docx import Document  # Optional dependency for .docx
except ImportError:
    Document = None


def clean_text(text: str) -> str:
    """
    Clean extra whitespace and control characters.

    Parameters
    ----------
    text : str
        Raw text content.

    Returns
    -------
    str
        Cleaned text with normalized whitespace.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_documents(paths: List[str]) -> List[Dict]:
    """
    Load text content from multiple file types.

    Parameters
    ----------
    paths : List[str]
        A list of file paths to load.

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
