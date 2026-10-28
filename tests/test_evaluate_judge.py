"""
Tests for the LLM-as-judge added to src/evaluate.py.

All tests use mocked Ollama responses; no running model is required.
"""
from unittest.mock import MagicMock

from src.evaluate import (
    LLMJudge,
    _compute_abstention_precision,
    _format_context_for_judge,
    _parse_judge_json,
    evaluate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ollama_client(content: str) -> MagicMock:
    """Return a mock Ollama client that always returns *content*."""
    client = MagicMock()
    client.chat.return_value = {"message": {"content": content}}
    return client


VALID_JSON_RESPONSE = """{
  "correctness": 1,
  "faithfulness": 0.9,
  "correctness_reason": "The answer matches the gold answer exactly.",
  "faithfulness_reason": "Every claim is supported by the context."
}"""

PARTIAL_JSON_RESPONSE = """{
  "correctness": 0.5,
  "faithfulness": 0.75,
  "correctness_reason": "Partial match.",
  "faithfulness_reason": "Most claims supported."
}"""

ZERO_SCORE_JSON_RESPONSE = """{
  "correctness": 0,
  "faithfulness": 0.0,
  "correctness_reason": "Answer is completely wrong.",
  "faithfulness_reason": "No claims supported by context."
}"""

REFUSAL_SCORE_RESPONSE = """{
  "correctness": 1,
  "faithfulness": 1.0,
  "correctness_reason": "Both gold and generated are refusals.",
  "faithfulness_reason": "A refusal makes no unsupported claims."
}"""

MALFORMED_JSON_RESPONSE = "Sorry, I cannot score this answer."

FENCED_JSON_RESPONSE = """```json
{
  "correctness": 0.5,
  "faithfulness": 0.6,
  "correctness_reason": "Partial.",
  "faithfulness_reason": "Mostly faithful."
}
```"""

OUT_OF_RANGE_RESPONSE = """{
  "correctness": 2,
  "faithfulness": 1.5,
  "correctness_reason": "x",
  "faithfulness_reason": "y"
}"""


SAMPLE_CHUNKS = [
    {
        "source_file": "rbi_doc.PDF",
        "pages": [1, 2],
        "text": "The minimum net worth requirement is Rs 200 crore.",
        "doc_id": "rbi_doc",
    },
    {
        "source_file": "rbi_doc2.PDF",
        "pages": [5],
        "text": "Scheduled commercial banks must maintain a CRR of 4%.",
        "doc_id": "rbi_doc2",
    },
]


# ---------------------------------------------------------------------------
# _parse_judge_json
# ---------------------------------------------------------------------------


class TestParseJudgeJson:
    def test_valid_json_parsed_correctly(self):
        result = _parse_judge_json(VALID_JSON_RESPONSE)
        assert result is not None
        assert result["correctness"] == 1.0
        assert result["faithfulness"] == 0.9
        assert "matches" in result["correctness_reason"].lower()

    def test_partial_score(self):
        result = _parse_judge_json(PARTIAL_JSON_RESPONSE)
        assert result is not None
        assert result["correctness"] == 0.5
        assert result["faithfulness"] == 0.75

    def test_zero_score(self):
        result = _parse_judge_json(ZERO_SCORE_JSON_RESPONSE)
        assert result is not None
        assert result["correctness"] == 0.0
        assert result["faithfulness"] == 0.0

    def test_malformed_json_returns_none(self):
        assert _parse_judge_json(MALFORMED_JSON_RESPONSE) is None

    def test_fenced_code_block_stripped(self):
        result = _parse_judge_json(FENCED_JSON_RESPONSE)
        assert result is not None
        assert result["correctness"] == 0.5
        assert result["faithfulness"] == 0.6

    def test_out_of_range_correctness_returns_none(self):
        assert _parse_judge_json(OUT_OF_RANGE_RESPONSE) is None

    def test_invalid_correctness_value_returns_none(self):
        bad = '{"correctness": 0.3, "faithfulness": 0.5, "correctness_reason": "x", "faithfulness_reason": "y"}'
        assert _parse_judge_json(bad) is None

    def test_empty_string_returns_none(self):
        assert _parse_judge_json("") is None

    def test_extra_preamble_text_handled(self):
        preamble = (
            "Sure, here is my evaluation:\n"
            '{"correctness": 1, "faithfulness": 0.8, '
            '"correctness_reason": "Good.", "faithfulness_reason": "Fine."}'
        )
        result = _parse_judge_json(preamble)
        assert result is not None
        assert result["correctness"] == 1.0
        assert result["faithfulness"] == 0.8

    def test_faithfulness_boundary_zero(self):
        data = '{"correctness": 0, "faithfulness": 0.0, "correctness_reason": "x", "faithfulness_reason": "y"}'
        result = _parse_judge_json(data)
        assert result is not None
        assert result["faithfulness"] == 0.0

    def test_faithfulness_boundary_one(self):
        data = '{"correctness": 1, "faithfulness": 1.0, "correctness_reason": "x", "faithfulness_reason": "y"}'
        result = _parse_judge_json(data)
        assert result is not None
        assert result["faithfulness"] == 1.0

    def test_faithfulness_out_of_range_negative(self):
        data = '{"correctness": 1, "faithfulness": -0.1, "correctness_reason": "x", "faithfulness_reason": "y"}'
        assert _parse_judge_json(data) is None

    def test_faithfulness_out_of_range_above_one(self):
        data = '{"correctness": 1, "faithfulness": 1.1, "correctness_reason": "x", "faithfulness_reason": "y"}'
        assert _parse_judge_json(data) is None


# ---------------------------------------------------------------------------
# _format_context_for_judge
# ---------------------------------------------------------------------------


class TestFormatContextForJudge:
    def test_empty_chunks_returns_placeholder(self):
        result = _format_context_for_judge([])
        assert "no context" in result.lower()

    def test_single_chunk_formatted(self):
        chunks = [{"source_file": "doc.PDF", "pages": [3], "text": "Some text."}]
        result = _format_context_for_judge(chunks)
        assert "[1] doc.PDF (Page 3):" in result
        assert "Some text." in result

    def test_multi_page_chunk_formatted(self):
        chunks = [{"source_file": "doc.PDF", "pages": [3, 5], "text": "Content."}]
        result = _format_context_for_judge(chunks)
        assert "Pages 3-5" in result

    def test_multiple_chunks_indexed(self):
        result = _format_context_for_judge(SAMPLE_CHUNKS)
        assert "[1]" in result
        assert "[2]" in result

    def test_missing_pages_shows_unknown(self):
        chunks = [{"source_file": "doc.PDF", "pages": [], "text": "text"}]
        result = _format_context_for_judge(chunks)
        assert "unknown page" in result


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class TestLLMJudge:
    def test_successful_scoring(self):
        client = _make_ollama_client(VALID_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        result = judge.judge(
            question="What is the CRR?",
            gold_answer="4%",
            generated_answer="The CRR is 4%.",
            retrieved_chunks=SAMPLE_CHUNKS,
        )
        assert result["judge_error"] is None
        assert result["judge_skipped"] is False
        assert result["correctness"] == 1.0
        assert result["faithfulness"] == 0.9

    def test_partial_score_returned(self):
        client = _make_ollama_client(PARTIAL_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        result = judge.judge(
            question="Q?", gold_answer="A", generated_answer="Partial A.", retrieved_chunks=[]
        )
        assert result["correctness"] == 0.5
        assert result["faithfulness"] == 0.75
        assert result["judge_error"] is None

    def test_malformed_json_is_flagged_as_error(self):
        client = _make_ollama_client(MALFORMED_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        result = judge.judge(
            question="Q?", gold_answer="A", generated_answer="B", retrieved_chunks=[]
        )
        assert result["correctness"] is None
        assert result["faithfulness"] is None
        assert result["judge_error"] is not None
        assert "malformed_json" in result["judge_error"]
        assert result["judge_skipped"] is False

    def test_ollama_exception_is_flagged_as_error(self):
        client = MagicMock()
        client.chat.side_effect = ConnectionError("Ollama not running")
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        result = judge.judge(
            question="Q?", gold_answer="A", generated_answer="B", retrieved_chunks=[]
        )
        assert result["correctness"] is None
        assert result["faithfulness"] is None
        assert result["judge_error"] is not None
        assert "ollama_call_failed" in result["judge_error"]
        assert result["judge_skipped"] is False

    def test_fenced_json_parsed_correctly(self):
        client = _make_ollama_client(FENCED_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        result = judge.judge(
            question="Q?", gold_answer="A", generated_answer="B", retrieved_chunks=[]
        )
        assert result["judge_error"] is None
        assert result["correctness"] == 0.5

    def test_refusal_gets_full_marks(self):
        client = _make_ollama_client(REFUSAL_SCORE_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        refusal_phrase = "Not found in the provided documents."
        result = judge.judge(
            question="What is the crypto staking rule?",
            gold_answer=refusal_phrase,
            generated_answer=refusal_phrase,
            retrieved_chunks=[],
        )
        assert result["correctness"] == 1.0
        assert result["faithfulness"] == 1.0
        assert result["judge_error"] is None

    def test_prompt_contains_question_and_gold_answer(self):
        """Verify that the Ollama call includes the question and gold answer in its prompt."""
        client = _make_ollama_client(VALID_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        judge.judge(
            question="What is the SLR rate?",
            gold_answer="18%",
            generated_answer="The SLR is 18%.",
            retrieved_chunks=[],
        )
        call_args = client.chat.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        assert "What is the SLR rate?" in prompt_text
        assert "18%" in prompt_text

    def test_temperature_zero_passed_to_ollama(self):
        client = _make_ollama_client(VALID_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", temperature=0.0, ollama_client=client)
        judge.judge(
            question="Q?", gold_answer="A", generated_answer="B", retrieved_chunks=[]
        )
        options = client.chat.call_args[1]["options"]
        assert options["temperature"] == 0.0

    def test_model_name_passed_to_ollama(self):
        client = _make_ollama_client(VALID_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        judge.judge(
            question="Q?", gold_answer="A", generated_answer="B", retrieved_chunks=[]
        )
        assert client.chat.call_args[1]["model"] == "qwen2.5:7b"


# ---------------------------------------------------------------------------
# _compute_abstention_precision
# ---------------------------------------------------------------------------


class TestComputeAbstentionPrecision:
    def test_no_refusals_returns_none(self):
        records = [{"is_refusal": False, "type": "simple_factual"}]
        assert _compute_abstention_precision(records) is None

    def test_all_refusals_on_unanswerable(self):
        records = [
            {"is_refusal": True, "type": "unanswerable"},
            {"is_refusal": True, "type": "unanswerable"},
        ]
        assert _compute_abstention_precision(records) == 1.0

    def test_no_refusals_on_unanswerable(self):
        records = [
            {"is_refusal": True, "type": "simple_factual"},
        ]
        assert _compute_abstention_precision(records) == 0.0

    def test_mixed_refusals(self):
        records = [
            {"is_refusal": True, "type": "unanswerable"},
            {"is_refusal": True, "type": "simple_factual"},
            {"is_refusal": False, "type": "unanswerable"},
        ]
        # 1 true positive out of 2 refusals
        assert _compute_abstention_precision(records) == 0.5

    def test_empty_records(self):
        assert _compute_abstention_precision([]) is None


# ---------------------------------------------------------------------------
# evaluate() integration — judge disabled (retrieval_only path) and enabled
# ---------------------------------------------------------------------------


class TestEvaluateWithJudge:
    """Integration tests using mocked pipeline and judge components."""

    def _make_pipeline(
        self,
        answer: str = "The CRR is 4%.",
        is_refusal: bool = False,
        retrieved_chunks: list[dict] | None = None,
        doc_id: str = "rbi_doc",
    ) -> MagicMock:
        from src.pipeline import RAGPipeline

        chunks = retrieved_chunks or [
            {"doc_id": doc_id, "source_file": "rbi_doc.PDF", "pages": [1], "text": "CRR is 4%."}
        ]
        pipeline = MagicMock(spec=RAGPipeline)
        pipeline.answer.return_value = {
            "answer": answer,
            "citations": [],
            "is_refusal": is_refusal,
            "retrieved_chunks": chunks,
        }
        pipeline.retriever = MagicMock()
        pipeline.retriever.retrieve.return_value = chunks
        return pipeline

    def _sample_questions(self) -> list[dict]:
        return [
            {
                "id": "Q01",
                "question": "What is the CRR?",
                "type": "simple_factual",
                "gold_answer": "4%",
                "gold_doc_ids": ["rbi_doc"],
                "answerable": True,
            },
            {
                "id": "Q02",
                "question": "What is the crypto staking rule?",
                "type": "unanswerable",
                "gold_answer": "Not found in the provided documents.",
                "gold_doc_ids": [],
                "answerable": False,
            },
        ]

    def test_judge_disabled_when_no_judge_passed(self):
        pipeline = self._make_pipeline()
        questions = self._sample_questions()
        report = evaluate(questions, pipeline, retrieval_only=False, judge=None)
        # All records should have judge_skipped=True
        for record in report["records"]:
            assert record["judge_skipped"] is True
            assert record["judge_correctness"] is None
            assert record["judge_faithfulness"] is None

    def test_judge_not_called_in_retrieval_only_mode(self):
        pipeline = self._make_pipeline()
        client = _make_ollama_client(VALID_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        questions = self._sample_questions()
        evaluate(questions, pipeline, retrieval_only=True, judge=judge)
        # Judge should never be called in retrieval-only mode
        client.chat.assert_not_called()

    def test_judge_called_for_answerable_questions(self):
        pipeline = self._make_pipeline(answer="4%", is_refusal=False, doc_id="rbi_doc")
        client = _make_ollama_client(VALID_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        questions = self._sample_questions()
        evaluate(questions, pipeline, retrieval_only=False, judge=judge)
        # Judge must have been called at least once (Q01 and Q02 both have gold answers)
        assert client.chat.call_count >= 1

    def test_judge_skipped_for_empty_gold_answer(self):
        pipeline = self._make_pipeline()
        client = _make_ollama_client(VALID_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        # Ambiguous questions have empty gold answers
        questions = [
            {
                "id": "Q35",
                "question": "What is the limit?",
                "type": "ambiguous",
                "gold_answer": "",
                "gold_doc_ids": [],
                "answerable": None,
            }
        ]
        report = evaluate(questions, pipeline, retrieval_only=False, judge=judge)
        # Judge must not be called for questions with empty gold_answer
        client.chat.assert_not_called()
        record = report["records"][0]
        assert record["judge_skipped"] is True
        assert record["judge_correctness"] is None

    def test_aggregate_metrics_populated_on_success(self):
        pipeline = self._make_pipeline(answer="4%", is_refusal=False, doc_id="rbi_doc")
        client = _make_ollama_client(VALID_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        questions = [self._sample_questions()[0]]  # Q01 only
        report = evaluate(questions, pipeline, retrieval_only=False, judge=judge)
        jm = report["metrics"]["judge"]
        assert jm["mean_correctness"] == 1.0
        assert jm["mean_faithfulness"] == 0.9
        assert jm["judge_failures"] == 0

    def test_judge_failure_counted_and_does_not_crash(self):
        pipeline = self._make_pipeline()
        client = MagicMock()
        client.chat.side_effect = RuntimeError("Model unavailable")
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        questions = [self._sample_questions()[0]]  # Q01 only
        # Must not raise
        report = evaluate(questions, pipeline, retrieval_only=False, judge=judge)
        jm = report["metrics"]["judge"]
        assert jm["judge_failures"] == 1
        assert jm["mean_correctness"] is None

    def test_existing_retrieval_metrics_unchanged_with_judge(self):
        pipeline = self._make_pipeline(doc_id="rbi_doc")
        client = _make_ollama_client(VALID_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        questions = [self._sample_questions()[0]]
        report = evaluate(questions, pipeline, retrieval_only=False, judge=judge)
        metrics = report["metrics"]
        # Retrieval metrics must still be present and valid
        assert "recall_at_5" in metrics
        assert "recall_at_10" in metrics
        assert "mrr_at_10" in metrics
        assert "exact_substring_correctness" in metrics
        assert "abstention_recall_on_unanswerable" in metrics

    def test_judge_metrics_absent_key_when_disabled(self):
        pipeline = self._make_pipeline()
        questions = self._sample_questions()
        report = evaluate(questions, pipeline, retrieval_only=False, judge=None)
        jm = report["metrics"]["judge"]
        assert jm["mean_correctness"] is None
        assert jm["mean_faithfulness"] is None
        assert jm["judged_questions"] == 0

    def test_hallucination_rate_computed(self):
        # Pipeline always refuses on unanswerable (is_refusal=True for Q02)
        from src.pipeline import RAGPipeline

        pipeline = MagicMock(spec=RAGPipeline)

        def _side_effect(prompt, top_k=10):
            if "crypto" in prompt.lower():
                return {
                    "answer": "Not found in the provided documents.",
                    "citations": [],
                    "is_refusal": True,
                    "retrieved_chunks": [],
                }
            return {
                "answer": "The CRR is 4%.",
                "citations": [],
                "is_refusal": False,
                "retrieved_chunks": [{"doc_id": "rbi_doc", "source_file": "x.PDF", "pages": [1], "text": "CRR 4%"}],
            }

        pipeline.answer.side_effect = _side_effect

        client = _make_ollama_client(VALID_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        questions = self._sample_questions()
        report = evaluate(questions, pipeline, retrieval_only=False, judge=judge)
        jm = report["metrics"]["judge"]
        # 1 unanswerable question, model refused → hallucination_rate = 0.0
        assert jm["hallucination_rate"] == 0.0

    def test_records_contain_judge_fields(self):
        pipeline = self._make_pipeline(answer="4%")
        client = _make_ollama_client(VALID_JSON_RESPONSE)
        judge = LLMJudge(model="qwen2.5:7b", ollama_client=client)
        questions = [self._sample_questions()[0]]
        report = evaluate(questions, pipeline, retrieval_only=False, judge=judge)
        record = report["records"][0]
        assert "judge_correctness" in record
        assert "judge_faithfulness" in record
        assert "judge_correctness_reason" in record
        assert "judge_faithfulness_reason" in record
        assert "judge_error" in record
        assert "judge_skipped" in record
