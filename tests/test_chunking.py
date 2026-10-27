"""Tests for document chunking, metadata preservation, and page mapping."""
import pytest

from src.chunking import (
    chunk_document_text,
    extract_page_markers,
    get_tokenizer,
    resolve_pages,
)


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer("BAAI/bge-base-en-v1.5")


def test_chunk_small_document_produces_single_chunk(tokenizer):
    text = "# Sample\n\n## Page 1\n\nReserve Bank of India guidelines on commercial paper."
    meta = {
        "doc_id": "doc123",
        "source_file": "doc123.PDF",
        "title": "Guidelines on Commercial Paper",
        "source_url": "https://rbi.org.in/sample",
    }
    chunks = chunk_document_text(
        text=text,
        tokenizer=tokenizer,
        chunk_size=512,
        chunk_overlap=64,
        doc_metadata=meta,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["chunk_id"] == "doc123_chunk_0"
    assert chunk["source_file"] == "doc123.PDF"
    assert chunk["title"] == "Guidelines on Commercial Paper"
    assert chunk["pages"] == [1]
    assert "guidelines on commercial paper" in chunk["text"].lower()
    assert 0 < chunk["token_count"] <= 512


def test_chunk_large_document_produces_overlapping_chunks(tokenizer):
    # Repeat text to produce > 1000 tokens
    paragraph = (
        "The Reserve Bank of India regulates and supervises non-banking financial companies "
        "and statutory liquidity ratios across scheduled commercial banks. "
    )
    long_text = "# Long Document\n\n## Page 1\n\n" + (paragraph * 70)

    chunks = chunk_document_text(
        text=long_text,
        tokenizer=tokenizer,
        chunk_size=512,
        chunk_overlap=64,
        doc_metadata={"doc_id": "large_doc"},
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert 0 < chunk["token_count"] <= 512
        assert chunk["text"].strip()  # No empty chunks
        assert chunk["source_file"] == "large_doc.PDF"

    assert chunks[1]["chunk_index"] == 1
    assert chunks[0]["chunk_index"] == 0


def test_empty_document_produces_no_chunks(tokenizer):
    assert chunk_document_text("", tokenizer, 512, 64) == []
    assert chunk_document_text("   \n\n  ", tokenizer, 512, 64) == []


def test_invalid_chunk_overlap_raises(tokenizer):
    with pytest.raises(ValueError, match="strictly greater"):
        chunk_document_text("sample", tokenizer, chunk_size=64, chunk_overlap=64)


def test_page_marker_extraction_and_resolution():
    text = (
        "# Doc\n\n"
        "## Page 1\n\nContent for page 1.\n\n"
        "## Page 2\n\nContent for page 2.\n\n"
        "## Page 3\n\nContent for page 3."
    )
    markers = extract_page_markers(text)
    assert len(markers) == 3
    assert markers[0][1] == 1
    assert markers[1][1] == 2
    assert markers[2][1] == 3

    # Span within page 1
    p1 = resolve_pages(markers, start_char=markers[0][0] + 5, end_char=markers[1][0] - 5)
    assert p1 == [1]

    # Span spanning page 1 into page 2
    p1_2 = resolve_pages(markers, start_char=markers[0][0] + 5, end_char=markers[1][0] + 10)
    assert p1_2 == [1, 2]

    # Span spanning across all 3 pages
    p1_3 = resolve_pages(markers, start_char=markers[0][0], end_char=len(text))
    assert p1_3 == [1, 2, 3]
