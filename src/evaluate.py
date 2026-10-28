"""Evaluation harness for the RBI document RAG system."""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from src.config import load_config
from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "qa_pairs.jsonl"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "baseline.json"

# ---------------------------------------------------------------------------
# Correctness scores the judge may return.
# ---------------------------------------------------------------------------
VALID_CORRECTNESS_SCORES = {0, 0.5, 1}

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------
_JUDGE_PROMPT_TEMPLATE = """\
You are a strict evaluator for a question-answering system built on Reserve Bank of India (RBI) documents.

Your task: given a question, a gold (reference) answer, a generated answer, and the retrieved context
passages, score the generated answer on two dimensions.

IMPORTANT RULES:
- Use ONLY the supplied gold answer and context passages as evidence.
- Do NOT use any outside knowledge. Do not penalise for information that is not in the context.
- Be consistent and objective.

---
QUESTION:
{question}

---
GOLD ANSWER (authoritative reference):
{gold_answer}

---
GENERATED ANSWER (to evaluate):
{generated_answer}

---
RETRIEVED CONTEXT PASSAGES:
{context}

---
SCORING DIMENSIONS:

1. CORRECTNESS  (compare generated answer to the gold answer)
   - 1   : The generated answer is fully correct and captures all key facts in the gold answer.
   - 0.5 : The generated answer is partially correct; it captures some key facts but is incomplete or
           has a minor error.
   - 0   : The generated answer is incorrect, unrelated, or a refusal when the gold answer is non-empty.
   Special case: if the gold answer is "Not found in the provided documents." and the generated answer
   is also a refusal, score correctness = 1.

2. FAITHFULNESS  (compare generated answer to the retrieved context)
   - A claim is faithful if it can be directly verified from the supplied context passages.
   - Score = (number of faithful claims) / (total number of factual claims in the generated answer).
   - If the generated answer makes no factual claims (e.g. it is a refusal), faithfulness = 1.
   - Range: 0.0 to 1.0 (two decimal places).

---
Respond with ONLY the following JSON object and nothing else:

{{
  "correctness": <0, 0.5, or 1>,
  "faithfulness": <0.0 to 1.0>,
  "correctness_reason": "<one sentence>",
  "faithfulness_reason": "<one sentence>"
}}
"""


def _format_context_for_judge(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a compact block for the judge prompt."""
    if not chunks:
        return "(no context retrieved)"

    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        source = chunk.get("source_file", "unknown")
        pages = chunk.get("pages", [])
        if pages:
            pages_str = (
                f"Page {pages[0]}" if len(pages) == 1
                else f"Pages {pages[0]}-{pages[-1]}"
            )
        else:
            pages_str = "unknown page"
        text = chunk.get("text", "").strip()
        blocks.append(f"[{idx}] {source} ({pages_str}):\n{text}")

    return "\n\n".join(blocks)


def _parse_judge_json(raw: str) -> dict[str, Any] | None:
    """
    Extract and parse the JSON object from a judge response string.

    Returns None when the response is malformed or values are out of range.
    """
    # Strip markdown code fences if the model added them.
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()

    # Try to find the first {...} block in case the model added preamble text.
    match = re.search(r"\{[^{}]+\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    # Validate required numeric fields.
    try:
        correctness = float(parsed.get("correctness", -1))
        faithfulness = float(parsed.get("faithfulness", -1))
    except (TypeError, ValueError):
        return None

    if correctness not in VALID_CORRECTNESS_SCORES:
        return None

    if not (0.0 <= faithfulness <= 1.0):
        return None

    return {
        "correctness": correctness,
        "faithfulness": round(faithfulness, 4),
        "correctness_reason": str(parsed.get("correctness_reason", "")).strip(),
        "faithfulness_reason": str(parsed.get("faithfulness_reason", "")).strip(),
    }


class LLMJudge:
    """
    Evaluates generated answers for correctness and faithfulness using a
    local Ollama model.  Never calls any external API.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        temperature: float = 0.0,
        ollama_client: Any = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = ollama_client

    def _call_ollama(self, prompt: str) -> str:
        """Send prompt to Ollama; return raw content string."""
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": self.temperature},
        }
        if self._client is not None:
            response = self._client.chat(**call_kwargs)
        else:
            import ollama  # type: ignore[import-untyped]

            response = ollama.chat(**call_kwargs)

        msg = response.get("message", {})
        return str(msg.get("content", "")).strip()

    def judge(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Score one question-answer pair.

        Returns a dict with keys:
            correctness         float | None
            faithfulness        float | None
            correctness_reason  str
            faithfulness_reason str
            judge_error         str | None   (populated on failure)
            judge_skipped       bool
        """
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
            context=_format_context_for_judge(retrieved_chunks),
        )

        try:
            raw = self._call_ollama(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Judge Ollama call failed for question %r: %s",
                question[:60],
                exc,
            )
            return {
                "correctness": None,
                "faithfulness": None,
                "correctness_reason": "",
                "faithfulness_reason": "",
                "judge_error": f"ollama_call_failed: {exc}",
                "judge_skipped": False,
            }

        parsed = _parse_judge_json(raw)
        if parsed is None:
            logger.warning(
                "Judge returned malformed JSON for question %r. Raw: %r",
                question[:60],
                raw[:200],
            )
            return {
                "correctness": None,
                "faithfulness": None,
                "correctness_reason": "",
                "faithfulness_reason": "",
                "judge_error": f"malformed_json: {raw[:200]}",
                "judge_skipped": False,
            }

        return {
            **parsed,
            "judge_error": None,
            "judge_skipped": False,
        }


# ---------------------------------------------------------------------------
# Existing helpers (unchanged)
# ---------------------------------------------------------------------------


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
                raise ValueError(  # noqa: TRY004
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


# ---------------------------------------------------------------------------
# Abstention precision helper
# ---------------------------------------------------------------------------


def _compute_abstention_precision(records: list[dict[str, Any]]) -> float | None:
    """
    Of all questions where the model issued a refusal, compute the fraction
    that were genuinely unanswerable (type == "unanswerable").
    """
    refusals = [r for r in records if r.get("is_refusal")]
    if not refusals:
        return None

    true_positives = sum(
        1 for r in refusals if r.get("type") == "unanswerable"
    )
    return round(true_positives / len(refusals), 4)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def evaluate(
    questions: list[dict[str, Any]],
    pipeline: RAGPipeline,
    retrieval_only: bool = False,
    judge: LLMJudge | None = None,
) -> dict[str, Any]:
    """
    Evaluate retrieval and optionally generate answers.

    If *judge* is not None and *retrieval_only* is False, every generated
    answer is scored for correctness and faithfulness.
    """
    records = []

    # Retrieval accumulators (unchanged)
    recall5_hits = 0
    recall10_hits = 0
    mrr_total = 0.0
    retrieval_labeled = 0

    # Answer / abstention accumulators (unchanged)
    answerable_total = 0
    answerable_correct = 0
    unanswerable_total = 0
    unanswerable_refused = 0

    total_seconds = 0.0

    # Judge accumulators
    judge_correctness_scores: list[float] = []
    judge_faithfulness_scores: list[float] = []
    judge_failures = 0
    judge_skipped_count = 0

    run_judge = (judge is not None) and (not retrieval_only)

    for number, question in enumerate(questions, 1):
        question_id = str(question["id"])
        prompt = question["question"]

        gold_ids = {
            str(doc_id)
            for doc_id in (question.get("gold_doc_ids") or [])
        }

        answerable = question.get("answerable")
        gold_answer = question.get("gold_answer", "")

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

        # ------------------------------------------------------------------
        # LLM Judge scoring
        # ------------------------------------------------------------------
        judge_result: dict[str, Any] = {"judge_skipped": True}

        if run_judge:
            generated_answer = result.get("answer", "")

            # Skip questions with no gold answer (e.g. ambiguous type).
            if not gold_answer.strip():
                judge_result = {
                    "correctness": None,
                    "faithfulness": None,
                    "correctness_reason": "",
                    "faithfulness_reason": "",
                    "judge_error": None,
                    "judge_skipped": True,
                }
                judge_skipped_count += 1
            else:
                judge_result = judge.judge(
                    question=prompt,
                    gold_answer=gold_answer,
                    generated_answer=generated_answer,
                    retrieved_chunks=retrieved,
                )

                if judge_result.get("judge_error"):
                    judge_failures += 1
                else:
                    c = judge_result.get("correctness")
                    f = judge_result.get("faithfulness")
                    if c is not None:
                        judge_correctness_scores.append(float(c))
                    if f is not None:
                        judge_faithfulness_scores.append(float(f))

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
            # Judge fields (None when judge is disabled or skipped)
            "judge_correctness": judge_result.get("correctness"),
            "judge_faithfulness": judge_result.get("faithfulness"),
            "judge_correctness_reason": judge_result.get("correctness_reason", ""),
            "judge_faithfulness_reason": judge_result.get("faithfulness_reason", ""),
            "judge_error": judge_result.get("judge_error"),
            "judge_skipped": judge_result.get("judge_skipped", True),
        }

        records.append(record)

        judge_summary = ""
        if run_judge and not judge_result.get("judge_skipped"):
            err = judge_result.get("judge_error")
            if err:
                judge_summary = " | judge=FAIL"
            else:
                c = judge_result.get("correctness")
                f = judge_result.get("faithfulness")
                judge_summary = f" | corr={c} faith={f}"

        print(
            f"[{number}/{len(questions)}] {question_id}: "
            f"{elapsed:.1f}s | "
            f"refusal={record['is_refusal']} | "
            f"retrieval_labeled={bool(gold_ids)}"
            f"{judge_summary}"
        )

    count = len(questions)

    # ------------------------------------------------------------------
    # Aggregate judge metrics
    # ------------------------------------------------------------------
    judged_count = len(judge_correctness_scores)
    judge_metrics: dict[str, Any]

    if run_judge:
        judge_metrics = {
            "judged_questions": judged_count,
            "judge_failures": judge_failures,
            "judge_skipped": judge_skipped_count,
            "mean_correctness": (
                round(sum(judge_correctness_scores) / judged_count, 4)
                if judged_count
                else None
            ),
            "mean_faithfulness": (
                round(sum(judge_faithfulness_scores) / judged_count, 4)
                if judged_count
                else None
            ),
            "hallucination_rate": (
                round(1.0 - unanswerable_refused / unanswerable_total, 4)
                if unanswerable_total
                else None
            ),
            "abstention_precision": _compute_abstention_precision(records),
        }
    else:
        judge_metrics = {
            "judged_questions": 0,
            "judge_failures": 0,
            "judge_skipped": count,
            "mean_correctness": None,
            "mean_faithfulness": None,
            "hallucination_rate": (
                round(1.0 - unanswerable_refused / unanswerable_total, 4)
                if unanswerable_total
                else None
            ),
            "abstention_precision": _compute_abstention_precision(records),
        }

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
        "judge": judge_metrics,
        "note": (
            "Retrieval metrics use document-level relevance from "
            "gold_doc_ids. exact_substring_correctness is a naive "
            "heuristic; prefer judge.mean_correctness for semantic "
            "scoring. Abstention recall is recall on unanswerable "
            "questions; abstention_precision is computed from all "
            "refusals regardless of question type."
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

    parser.add_argument(
        "--no-judge",
        action="store_true",
        help=(
            "Disable LLM-as-judge scoring. "
            "By default the judge runs on every full-RAG evaluation."
        ),
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

    use_judge = (not args.no_judge) and (not args.retrieval_only)

    print(f"Questions loaded: {len(questions)}")
    print(
        "Evaluation mode: "
        f"{'retrieval-only' if args.retrieval_only else 'full RAG'}"
    )
    print(
        "Reranking: "
        f"{'enabled' if args.rerank else 'disabled'}"
    )
    print(
        "LLM judge: "
        f"{'enabled' if use_judge else 'disabled'}"
    )
    print("Loading RAG pipeline and local models...")

    cfg = load_config()
    pipeline = RAGPipeline(
        cfg,
        use_reranker=args.rerank,
    )

    judge: LLMJudge | None = None
    if use_judge:
        print(f"Judge model: {cfg.models.judge}")
        judge = LLMJudge(model=cfg.models.judge, temperature=0.0)

    report = evaluate(
        questions,
        pipeline,
        retrieval_only=args.retrieval_only,
        judge=judge,
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
