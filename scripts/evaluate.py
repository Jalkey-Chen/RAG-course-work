import json
from pathlib import Path

def load_questions(levels: list[str] = ["conceptual", "applied", "project"]) -> list[str]:
    """
    Load benchmark questions from the JSON file by category.
    """
    json_path = Path(__file__).parent / "eval_questions.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for lvl in levels:
        if lvl in data:
            questions.extend(data[lvl]["questions"])
    return questions


if __name__ == "__main__":
    print("Loading benchmark questions...")
    BENCHMARK_QUESTIONS = load_questions()
    for q in BENCHMARK_QUESTIONS:
        print("-", q)
