"""Tests for generation, refusal enforcement, prompt structure, and citations."""
from unittest.mock import MagicMock

from src.generation import (
    Generator,
    build_rag_prompt,
    estimate_token_count,
    extract_citations_from_chunks,
    format_context_chunks,
    is_refusal_response,
    select_context_chunks,
)


def test_format_context_chunks():
    chunks = [
        {
            "source_file": "rbi_doc1.PDF",
            "pages": [1, 2],
            "text": "First context block text.",
        },
        {
            "source_file": "rbi_doc2.PDF",
            "pages": [5],
            "text": "Second context block text.",
        },
    ]
    formatted = format_context_chunks(chunks)
    assert "Source Document: rbi_doc1.PDF (Pages 1-2)" in formatted
    assert "First context block text." in formatted
    assert "Source Document: rbi_doc2.PDF (Page 5)" in formatted
    assert "Second context block text." in formatted


def test_build_rag_prompt_contains_rules_and_refusal_phrase():
    prompt = build_rag_prompt(
        query="What is the SLR rate?",
        context="Sample context",
        refusal_phrase="Not found in the provided documents.",
    )
    assert "Not found in the provided documents." in prompt
    assert "What is the SLR rate?" in prompt
    assert "Sample context" in prompt
    assert "Strict Refusal Rule:" in prompt


def test_generator_with_valid_answer():
    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "message": {
            "content": "The minimum capital requirement is Rs 200 crore. [rbi_sfb.PDF, Page 1]"
        }
    }

    generator = Generator(
        model="qwen2.5:7b",
        temperature=0.0,
        refusal="Not found in the provided documents.",
        ollama_client=mock_client,
    )

    chunks = [
        {
            "source_file": "rbi_sfb.PDF",
            "pages": [1],
            "text": "Small finance bank minimum capital requirement is Rs 200 crore.",
        }
    ]

    result = generator.generate("What is the minimum capital requirement?", retrieved_chunks=chunks)

    assert result["is_refusal"] is False
    assert "Rs 200 crore" in result["answer"]
    assert result["citations"] == ["rbi_sfb.PDF (Page 1)"]

    # Verify Ollama options: temperature=0
    mock_client.chat.assert_called_once()
    call_kwargs = mock_client.chat.call_args[1]
    assert call_kwargs["options"]["temperature"] == 0.0
    assert call_kwargs["model"] == "qwen2.5:7b"


def test_generator_refusal_when_unsupported_by_context():
    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "message": {"content": "Not found in the provided documents."}
    }

    generator = Generator(
        model="qwen2.5:7b",
        temperature=0.0,
        refusal="Not found in the provided documents.",
        ollama_client=mock_client,
    )

    chunks = [
        {
            "source_file": "rbi_irrelevant.PDF",
            "pages": [2],
            "text": "Irrelevant text about foreign exchange branches.",
        }
    ]

    result = generator.generate("What is the inflation target for 2029?", retrieved_chunks=chunks)

    assert result["is_refusal"] is True
    assert result["answer"] == "Not found in the provided documents."
    assert result["citations"] == []


def test_generator_refusal_when_no_chunks():
    generator = Generator(
        model="qwen2.5:7b",
        temperature=0.0,
        refusal="Not found in the provided documents.",
    )
    result = generator.generate("Any question", retrieved_chunks=[])
    assert result["is_refusal"] is True
    assert result["answer"] == "Not found in the provided documents."
    assert result["citations"] == []


def test_extract_citations_unique():
    chunks = [
        {"source_file": "docA.PDF", "pages": [1]},
        {"source_file": "docA.PDF", "pages": [1]},
        {"source_file": "docB.PDF", "pages": [3, 4]},
    ]
    citations = extract_citations_from_chunks(chunks)
    assert len(citations) == 2
    assert "docA.PDF (Page 1)" in citations
    assert "docB.PDF (Pages 3-4)" in citations


def test_is_refusal_response_matching():
    target = "Not found in the provided documents."
    assert is_refusal_response("Not found in the provided documents.", target) is True
    assert is_refusal_response("  Not found in the provided documents.  ", target) is True
    assert is_refusal_response("'Not found in the provided documents.'", target) is True
    assert is_refusal_response("According to RBI, the interest rate is 6.5%.", target) is False


def test_estimate_token_count():
    assert estimate_token_count("") == 0
    assert estimate_token_count("Hello world") >= 2


def test_select_context_chunks_budgeting():
    # Create 5 chunks, each with ~100 tokens of text
    chunks = [
        {
            "source_file": f"doc_{i}.pdf",
            "pages": [i],
            "text": f"This is chunk number {i} containing important financial regulations. " * 5,
        }
        for i in range(1, 6)
    ]

    # Without budget, all chunks returned
    assert len(select_context_chunks(chunks, max_tokens=None)) == 5

    # With a tight budget (e.g. 150 tokens), only top chunk(s) should be selected
    selected = select_context_chunks(chunks, max_tokens=150)
    assert 1 <= len(selected) <= 2
    # Ensure rank 1 chunk is prioritized
    assert selected[0]["source_file"] == "doc_1.pdf"


def test_format_context_chunks_with_token_budget():
    chunks = [
        {
            "source_file": f"doc_{i}.pdf",
            "pages": [i],
            "text": f"Important regulatory clause for doc {i}. " * 10,
        }
        for i in range(1, 6)
    ]
    formatted = format_context_chunks(chunks, max_tokens=200)
    assert "[Context 1] Source Document: doc_1.pdf" in formatted
    # doc_5 should not be in the tightly budgeted context
    assert "doc_5.pdf" not in formatted

