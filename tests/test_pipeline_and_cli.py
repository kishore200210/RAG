"""Tests for end-to-end RAG pipeline and CLI invocation."""
from unittest.mock import MagicMock, patch

from src.config import load_config
from src.generation import Generator
from src.pipeline import RAGPipeline, format_qa_output
from src.retrieval import Retriever


def test_pipeline_with_mocked_components():
    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.retrieve.return_value = [
        {
            "chunk_id": "c1",
            "source_file": "sample_rbi.PDF",
            "pages": [1],
            "text": "RBI prescribes capital requirements.",
            "score": 0.95,
        }
    ]

    mock_generator = MagicMock(spec=Generator)
    mock_generator.generate.return_value = {
        "query": "What are the capital requirements?",
        "answer": "RBI prescribes capital requirements. [sample_rbi.PDF, Page 1]",
        "citations": ["sample_rbi.PDF (Page 1)"],
        "is_refusal": False,
        "raw_response": "RBI prescribes capital requirements. [sample_rbi.PDF, Page 1]",
    }

    pipeline = RAGPipeline(
        config=load_config(),
        retriever=mock_retriever,
        generator=mock_generator,
    )

    result = pipeline.answer("What are the capital requirements?", top_k=1)
    assert result["is_refusal"] is False
    assert "capital requirements" in result["answer"]
    assert result["citations"] == ["sample_rbi.PDF (Page 1)"]
    assert len(result["retrieved_chunks"]) == 1

    formatted = format_qa_output(result, include_chunks=True)
    assert "Question: What are the capital requirements?" in formatted
    assert "Answer: RBI prescribes capital requirements." in formatted
    assert "Sources / Citations:" in formatted
    assert "sample_rbi.PDF (Page 1)" in formatted


def test_pipeline_empty_question_refusal():
    pipeline = RAGPipeline(
        config=load_config(),
        retriever=MagicMock(),
        generator=MagicMock(),
    )
    result = pipeline.answer("")
    assert result["is_refusal"] is True
    assert result["answer"] == "Not found in the provided documents."
    assert result["citations"] == []


def test_cli_help(capsys):
    from src.cli import build_parser

    parser = build_parser()
    with patch("sys.argv", ["rag-cli", "--help"]):
        try:
            parser.parse_args(["--help"])
        except SystemExit:
            pass
    captured = capsys.readouterr()
    assert "RBI Document Question-Answering" in captured.out or "index" in captured.out
