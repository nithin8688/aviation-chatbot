import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))



import json
from pathlib import Path
from src.bot import ask_aviation_bot

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_qa.json"


def keyword_match_score(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0 if "outside the aviation domain" in answer.lower() else 0.0

    answer_lower = answer.lower()
    matches = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(matches / len(keywords), 2)


def run_evaluation():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    total = len(golden_data)
    scores = []

    print("\n=== AVIATION CHATBOT EVALUATION ===\n")

    for item in golden_data:
        question = item["question"]
        expected_keywords = item["expected_keywords"]

        print(f"Q: {question}")
        answer = ask_aviation_bot(question, top_k=1)
        print(f"A: {answer}")

        score = keyword_match_score(answer, expected_keywords)
        scores.append(score)

        print(f"Score: {score}")
        print("-" * 80)

    avg_score = round(sum(scores) / total, 2)

    print("\n=== SUMMARY ===")
    print(f"Total Questions: {total}")
    print(f"Average Keyword Match Score: {avg_score}")

    if avg_score >= 0.7:
        print("Status: ✅ GOOD")
    elif avg_score >= 0.5:
        print("Status: ⚠️ NEEDS IMPROVEMENT")
    else:
        print("Status: ❌ POOR")


if __name__ == "__main__":
    run_evaluation()
