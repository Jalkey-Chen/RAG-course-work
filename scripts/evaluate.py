"""
scripts/evaluate.py
-------------------
Lightweight evaluation script for the RAG FastAPI backend.
It queries several benchmark questions and records answer stats.
"""

import requests
import time
import csv
from pathlib import Path


API_URL = "http://127.0.0.1:8000/api/query"
BENCHMARK_QUESTIONS = [
    "What does this dataset describe?",
    "What methodology is discussed?",
    "Who is the main author mentioned?",
    "What are the key findings about RAG?",
    "Summarize the main conclusion.",
]


def query_backend(question: str, k: int = 3) -> dict:
    """Send a POST request to the RAG FastAPI endpoint."""
    resp = requests.post(API_URL, json={"question": question, "k": k})
    resp.raise_for_status()
    return resp.json()


def evaluate(questions: list[str]) -> list[dict]:
    """
    Evaluate a batch of questions and compute basic metrics.

    Returns
    -------
    List[Dict]
        Each record includes question, token length, num_sources, duration, hallucination flag.
    """
    results = []
    for q in questions:
        start = time.time()
        try:
            result = query_backend(q)
            duration = round(time.time() - start, 2)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            hallucination = int("[" not in answer)  # crude heuristic

            results.append({
                "question": q,
                "chars": len(answer),
                "sources": len(sources),
                "duration": duration,
                "hallucination": hallucination,
            })
            print(f"✅ {q[:40]}... ({duration}s)")
        except Exception as e:
            print(f"❌ Error on '{q}': {e}")
    return results


def save_csv(records: list[dict], path: str = "evaluation_results.csv"):
    """Write results to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\n📊 Results saved to {path}")


if __name__ == "__main__":
    Path("scripts").mkdir(exist_ok=True)
    print("Running evaluation...")
    data = evaluate(BENCHMARK_QUESTIONS)
    if data:
        save_csv(data)
