"""
evaluate.py (v4)
----------------
RAG backend evaluator that saves full answers and annotates in-text citations
with source filenames, and now also includes retrieval scores.

What's new in v4
----------------
- Adds 'source_map_with_scores' column, e.g., "C1=foo.pdf (0.0312); C2=bar.html (0.1123)".
- Keeps:
  * 'answer' (original), 'answer_annotated' (with filenames),
  * 'source_files', 'source_map', plus timing & hallucination metrics.
- Works with eval_questions.json, CSV output, optional Markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import re
from pathlib import Path
from typing import Dict, List

import requests


DEFAULT_API_URL = "http://127.0.0.1:8000/api/query"
DEFAULT_JSON_PATH = Path(__file__).parent / "eval_questions.json"
DEFAULT_CATEGORIES = ["conceptual", "applied", "project"]
DEFAULT_CSV = Path("evaluation_results.csv")
DEFAULT_MD = Path("evaluation_results.md")


def load_questions(json_path: Path, categories: List[str]) -> List[str]:
    """Load benchmark questions from a JSON file and selected categories."""
    if not json_path.exists():
        raise FileNotFoundError(f"Question file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions: List[str] = []
    for cat in categories:
        block = data.get(cat)
        if not block:
            print(f"[WARN] Category '{cat}' not found in {json_path.name}; skipping.")
            continue
        qs = block.get("questions", [])
        if not isinstance(qs, list):
            print(f"[WARN] Malformed 'questions' under '{cat}'; skipping.")
            continue
        questions.extend(qs)

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for q in questions:
        if q not in seen:
            deduped.append(q)
            seen.add(q)

    if not deduped:
        raise ValueError("No questions loaded. Check categories or JSON content.")
    return deduped


def query_backend(api_url: str, question: str, k: int = 3, timeout: float = 120.0) -> Dict:
    """Send a POST request to the RAG FastAPI endpoint and return JSON."""
    payload = {"question": question, "k": k}
    r = requests.post(api_url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def maybe_truncate(text: str, limit: int) -> str:
    """Optionally truncate text to a maximum number of characters."""
    if limit and limit > 0 and len(text) > limit:
        return text[:limit]
    return text


def build_source_maps(sources: List[Dict]) -> tuple[Dict[str, str], str, str]:
    """
    Build:
      - ci_map:       {"C1": "file.pdf", ...}
      - source_map:   "C1=file.pdf; C2=note.html; ..."
      - source_map_ws:"C1=file.pdf (0.0312); C2=note.html (0.2210); ..."

    Notes
    -----
    - Expects each 'source' item to be a dict with keys:
      'metadata': {'source': '<path>'}, 'score': float
    - Scores are formatted to 4 decimal places.
    """
    ci_map: Dict[str, str] = {}
    parts: List[str] = []
    parts_ws: List[str] = []

    for i, s in enumerate(sources, start=1):
        meta = (s.get("metadata", {}) or {})
        src = meta.get("source", "unknown")
        fname = Path(src).name
        key = f"C{i}"
        ci_map[key] = fname

        score = s.get("score", None)
        parts.append(f"{key}={fname}")
        if score is not None:
            parts_ws.append(f"{key}={fname} ({float(score):.4f})")
        else:
            parts_ws.append(f"{key}={fname} (n/a)")

    return ci_map, "; ".join(parts), "; ".join(parts_ws)


def annotate_answer_with_sources(answer: str, ci_map: Dict[str, str]) -> str:
    """
    Replace occurrences of [C1], [C2], ... with [C1: filename] using ci_map.

    Uses a regex to catch "[C<number>]" patterns and preserves punctuation.
    """
    if not answer or not ci_map:
        return answer

    pattern = re.compile(r"\[C(\d+)\]")

    def _repl(match: re.Match) -> str:
        num = match.group(1)
        key = f"C{num}"
        if key in ci_map:
            return f"[{key}: {ci_map[key]}]"
        return match.group(0)

    return pattern.sub(_repl, answer)


def evaluate_questions(
    api_url: str,
    questions: List[str],
    k: int,
    max_answer_chars: int = 0
) -> List[Dict]:
    """
    Evaluate a list of questions against the backend.

    Recorded columns
    ----------------
    - question
    - answer                      (original)
    - answer_annotated            (with [C#] -> [C#: filename])
    - source_files                (comma-separated basenames)
    - source_map                  ("C1=foo.pdf; C2=bar.html; ...")
    - source_map_with_scores      ("C1=foo.pdf (0.0312); C2=bar.html (0.1123); ...")
    - chars                       (len(answer_annotated) after truncation)
    - sources                     (number of retrieved sources)
    - duration                    (seconds)
    - hallucination               (1 if no "[C" marker in original answer else 0)
    """
    rows: List[Dict] = []
    for q in questions:
        start = time.time()
        try:
            resp = query_backend(api_url, q, k=k)
            duration = round(time.time() - start, 2)

            answer_orig = (resp.get("answer", "") or "").strip()
            sources = resp.get("sources", []) or []

            # Build citation maps and annotate
            ci_map, src_map, src_map_ws = build_source_maps(sources)
            answer_annot = annotate_answer_with_sources(answer_orig, ci_map)

            # Truncation (applied to both answer fields for CSV size control)
            answer_orig_out = maybe_truncate(answer_orig, max_answer_chars)
            answer_annot_out = maybe_truncate(answer_annot, max_answer_chars)

            source_files_csv = ", ".join(
                Path((s.get("metadata", {}) or {}).get("source", "unknown")).name for s in sources
            )
            hallucination = 0 if "[C" in answer_orig else 1

            row = {
                "question": q,
                "answer": answer_orig_out,
                "answer_annotated": answer_annot_out,
                "source_files": source_files_csv,
                "source_map": src_map,
                "source_map_with_scores": src_map_ws,
                "chars": len(answer_annot_out),
                "sources": len(sources),
                "duration": duration,
                "hallucination": hallucination,
            }
            rows.append(row)
            print(f"✅ {q[:60]}... | t={duration}s, src={len(sources)}, halluc={hallucination}")
        except Exception as e:
            duration = round(time.time() - start, 2)
            print(f"❌ ERROR for '{q[:60]}...': {e} (t={duration}s)")
            rows.append({
                "question": q,
                "answer": "",
                "answer_annotated": "",
                "source_files": "",
                "source_map": "",
                "source_map_with_scores": "",
                "chars": 0,
                "sources": 0,
                "duration": duration,
                "hallucination": 1,
            })
    return rows


def save_csv(rows: List[Dict], out_csv: Path) -> None:
    """Save evaluation rows to a CSV file."""
    if not rows:
        print("[WARN] No rows to save; skipping CSV.")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"📊 CSV saved to: {out_csv.resolve()}")


def write_markdown_summary(rows: List[Dict], out_md: Path, categories: List[str], k: int) -> None:
    """Write a simple Markdown summary (averages + preview rows)."""
    if not rows:
        print("[WARN] No rows to summarize; skipping Markdown.")
        return

    n = len(rows)
    avg_time = sum(r["duration"] for r in rows) / n
    avg_chars = sum(r["chars"] for r in rows) / n
    avg_sources = sum(r["sources"] for r in rows) / n
    halluc_rate = sum(r["hallucination"] for r in rows) / n

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# RAG Evaluation Summary\n\n")
        f.write(f"- Categories: {', '.join(categories)}\n")
        f.write(f"- Top-k: {k}\n")
        f.write(f"- Total questions: {n}\n\n")
        f.write("## Averages\n\n")
        f.write(f"- Avg. response time (s): {avg_time:.2f}\n")
        f.write(f"- Avg. answer length (chars): {avg_chars:.1f}\n")
        f.write(f"- Avg. sources used: {avg_sources:.2f}\n")
        f.write(f"- Hallucination rate: {halluc_rate:.2%}\n\n")

        f.write("## Sample Rows (first 5)\n\n")
        f.write("| question | sources | duration | hallucination |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in rows[:5]:
            q_short = (r["question"][:60] + "...") if len(r["question"]) > 60 else r["question"]
            f.write(f"| {q_short} | {r['sources']} | {r['duration']:.2f} | {r['hallucination']} |\n")

        f.write("\n> Note: See the CSV for full answers and annotated citations with scores.\n")

    print(f"📝 Markdown summary saved to: {out_md.resolve()}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script."""
    p = argparse.ArgumentParser(description="Evaluate RAG backend and annotate answers with source filenames and scores.")
    p.add_argument("--api", type=str, default=DEFAULT_API_URL, help=f"Query endpoint URL (default: {DEFAULT_API_URL})")
    p.add_argument("--json", type=str, default=str(DEFAULT_JSON_PATH), help=f"Path to eval_questions.json (default: {DEFAULT_JSON_PATH})")
    p.add_argument("--categories", type=str, default=",".join(DEFAULT_CATEGORIES), help="Comma-separated categories (e.g., conceptual,applied,project).")
    p.add_argument("--k", type=int, default=3, help="Top-k contexts to retrieve per query (default: 3).")
    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help=f"Output CSV path (default: {DEFAULT_CSV}).")
    p.add_argument("--markdown", action="store_true", help=f"Also write a Markdown summary to {DEFAULT_MD}.")
    p.add_argument("--max-answer-chars", type=int, default=0, help="If > 0, truncate 'answer' and 'answer_annotated' in CSV.")
    return p.parse_args()


def main() -> None:
    """Script entrypoint: load questions, evaluate, and save results."""
    args = parse_args()

    json_path = Path(args.json)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    out_csv = Path(args.csv)

    print(f"[INFO] Loading questions from: {json_path}")
    print(f"[INFO] Categories: {categories}")
    questions = load_questions(json_path, categories)

    print(f"[INFO] Evaluating {len(questions)} questions against {args.api} (k={args.k})")
    rows = evaluate_questions(args.api, questions, k=args.k, max_answer_chars=args.max_answer_chars)

    save_csv(rows, out_csv)

    if args.markdown:
        write_markdown_summary(rows, DEFAULT_MD, categories, args.k)


if __name__ == "__main__":
    main()
