
"""Evaluation harness for the RBI document RAG system."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.config import load_config
from src.pipeline import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "qa_pairs.jsonl"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "baseline.json"


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load and validate one JSON object per line."""
    questions = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            if not row.get("question") or not isinstance(
                row["question"], str
            ):
                raise ValueError(
                    f"Line {line_number} has no valid question."
                )

            if not row.get("id"):
                raise ValueError(
                    f"Line {line_number} has no question ID."
                )

            questions.append(row)

    if not questions:
        raise ValueError(f"No questions found in {path}")

    return questions


def reciprocal_rank(
    retrieved_ids: list[str],
    gold_ids: set[str],
) -> float:
    """Return reciprocal rank of the first relevant retrieved document."""
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in gold_ids:
            return 1.0 / rank

    return 0.0


def evaluate(
    questions: list[dict[str, Any]],
    pipeline: RAGPipeline,
    retrieval_only: bool = False,
) -> dict[str, Any]:
    """Evaluate retrieval and optionally generate answers."""
    records = []

    recall5_hits = 0
    recall10_hits = 0
    mrr_total = 0.0
    retrieval_labeled = 0

    answerable_total = 0
    answerable_correct = 0
    unanswerable_total = 0
    unanswerable_refused = 0

    total_seconds = 0.0

    for number, question in enumerate(questions, 1):
        question_id = str(question["id"])
        prompt = question["question"]

        gold_ids = {
            str(doc_id)
            for doc_id in (question.get("gold_doc_ids") or [])
        }

        answerable = question.get("answerable")

        started = time.perf_counter()

        if retrieval_only:
            retrieved = pipeline.retriever.retrieve(
                prompt,
                top_k=10,
            )

            result = {
                "answer": "",
                "citations": [],
                "is_refusal": False,
                "retrieved_chunks": retrieved,
            }
        else:
            result = pipeline.answer(prompt, top_k=10)
            retrieved = result.get("retrieved_chunks", [])

        elapsed = time.perf_counter() - started
        total_seconds += elapsed

        retrieved = result.get("retrieved_chunks", [])

        retrieved_ids = [
            str(chunk.get("doc_id", ""))
            for chunk in retrieved
        ]

        relevant_found = bool(
            gold_ids.intersection(retrieved_ids)
        )

        # Retrieval metrics only use questions with gold document labels.
        if gold_ids:
            retrieval_labeled += 1

            recall5_hits += bool(
                gold_ids.intersection(retrieved_ids[:5])
            )

            recall10_hits += relevant_found

            mrr_total += reciprocal_rank(
                retrieved_ids,
                gold_ids,
            )

        # Answer and abstention metrics require generation.
        if not retrieval_only:
            if answerable is True:
                answerable_total += 1

                expected = (
                    question.get("gold_answer", "")
                    .strip()
                    .lower()
                )

                actual = (
                    result.get("answer", "")
                    .strip()
                    .lower()
                )

                if expected and expected in actual:
                    answerable_correct += 1

            elif answerable is False:
                unanswerable_total += 1

                if result.get("is_refusal"):
                    unanswerable_refused += 1

        record = {
            "id": question_id,
            "question": prompt,
            "type": question.get("type", "unspecified"),
            "gold_doc_ids": sorted(gold_ids),
            "retrieved_doc_ids": retrieved_ids,
            "relevant_document_retrieved": (
                relevant_found if gold_ids else None
            ),
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "is_refusal": result.get("is_refusal", False),
            "latency_seconds": round(elapsed, 3),
        }

        records.append(record)

        print(
            f"[{number}/{len(questions)}] {question_id}: "
            f"{elapsed:.1f}s | "
            f"refusal={record['is_refusal']} | "
            f"retrieval_labeled={bool(gold_ids)}"
        )

    count = len(questions)

    metrics = {
        "question_count": count,
        "retrieval_labeled_questions": retrieval_labeled,
        "recall_at_5": (
            recall5_hits / retrieval_labeled
            if retrieval_labeled
            else None
        ),
        "recall_at_10": (
            recall10_hits / retrieval_labeled
            if retrieval_labeled
            else None
        ),
        "mrr_at_10": (
            mrr_total / retrieval_labeled
            if retrieval_labeled
            else None
        ),
        "answerable_questions": answerable_total,
        "exact_substring_correctness": (
            answerable_correct / answerable_total
            if answerable_total
            else None
        ),
        "unanswerable_questions": unanswerable_total,
        "abstention_recall_on_unanswerable": (
            unanswerable_refused / unanswerable_total
            if unanswerable_total
            else None
        ),
        "average_latency_seconds": (
            total_seconds / count if count else 0.0
        ),
        "evaluation_mode": (
            "retrieval_only" if retrieval_only else "full_rag"
        ),
        "note": (
            "Retrieval metrics use document-level relevance from "
            "gold_doc_ids. Answer correctness uses exact-substring "
            "matching and requires manual validation; it is not a "
            "semantic correctness score. Abstention recall is measured "
            "only in full RAG mode and is recall on unanswerable "
            "questions, not precision."
        ),
    }

    return {
        "metrics": metrics,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the evaluation JSONL dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Path where evaluation results will be saved.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions.",
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking.",
    )

    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate retrieval without generating answers.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    if args.rerank and not args.retrieval_only:
        print(
            "Note: reranking is enabled; full answer generation "
            "will run for every question."
        )

    questions = load_questions(args.dataset)

    if args.limit is not None:
        questions = questions[:args.limit]

    print(f"Questions loaded: {len(questions)}")
    print(
        "Evaluation mode: "
        f"{'retrieval-only' if args.retrieval_only else 'full RAG'}"
    )
    print(
        "Reranking: "
        f"{'enabled' if args.rerank else 'disabled'}"
    )
    print("Loading RAG pipeline and local models...")

    pipeline = RAGPipeline(
        load_config(),
        use_reranker=args.rerank,
    )

    report = evaluate(
        questions,
        pipeline,
        retrieval_only=args.retrieval_only,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nMetrics:")
    print(json.dumps(report["metrics"], indent=2))

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
