import numpy as np

from src.index import (
    build_faiss_index,
    index_exists,
    load_index_and_metadata,
)
from src.retrieval import Retriever


class DummySentenceTransformer:
    """Fast dummy embedding model for testing without computing full neural embeddings."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            # Deterministic hash-based mock embedding
            for char in text:
                vec[ord(char) % self.dim] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def test_build_and_load_index_persistence(tmp_path):
    chunks = [
        {
            "chunk_id": "doc1_0",
            "doc_id": "doc1",
            "source_file": "doc1.PDF",
            "title": "Document 1",
            "pages": [1],
            "text": "The Reserve Bank of India regulates monetary policy.",
            "token_count": 10,
        },
        {
            "chunk_id": "doc2_0",
            "doc_id": "doc2",
            "source_file": "doc2.PDF",
            "title": "Document 2",
            "pages": [2, 3],
            "text": "Foreign exchange management act governs external payments.",
            "token_count": 10,
        },
    ]

    dummy_model = DummySentenceTransformer(dim=8)
    index, _indexed_chunks = build_faiss_index(
        chunks=chunks,
        model_name_or_instance=dummy_model,
        index_dir=tmp_path,
        show_progress_bar=False,
    )

    assert index.ntotal == 2
    assert index_exists(tmp_path)

    # Reload from disk without re-embedding
    loaded_index, loaded_chunks = load_index_and_metadata(tmp_path)
    assert loaded_index.ntotal == 2
    assert len(loaded_chunks) == 2
    assert loaded_chunks[0]["source_file"] == "doc1.PDF"
    assert loaded_chunks[1]["source_file"] == "doc2.PDF"


def test_retriever_returns_top_k_with_metadata():
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "doc1",
            "source_file": "rbi_monetary.PDF",
            "title": "Monetary Policy",
            "pages": [1],
            "text": "Monetary policy committee fixes benchmark repo rate.",
        },
        {
            "chunk_id": "c2",
            "doc_id": "doc2",
            "source_file": "rbi_banking.PDF",
            "title": "Banking Supervision",
            "pages": [4],
            "text": "Commercial bank capital adequacy framework under Basel III.",
        },
        {
            "chunk_id": "c3",
            "doc_id": "doc3",
            "source_file": "rbi_fema.PDF",
            "title": "Foreign Exchange",
            "pages": [1, 2],
            "text": "External commercial borrowings and rupee lending rules.",
        },
    ]

    dummy_model = DummySentenceTransformer(dim=8)
    index, _ = build_faiss_index(chunks=chunks, model_name_or_instance=dummy_model)
    retriever = Retriever(index=index, chunks=chunks, embedding_model=dummy_model, top_k=2)

    results = retriever.retrieve("repo rate monetary policy", top_k=2)
    assert len(results) == 2

    # Check schema
    for r in results:
        assert "chunk_id" in r
        assert "source_file" in r
        assert "pages" in r
        assert "text" in r
        assert "score" in r
        assert isinstance(r["score"], float)


def test_retriever_empty_query():
    dummy_model = DummySentenceTransformer(dim=8)
    chunks = [{"chunk_id": "c1", "text": "sample", "source_file": "sample.PDF"}]
    index, _ = build_faiss_index(chunks=chunks, model_name_or_instance=dummy_model)
    retriever = Retriever(index=index, chunks=chunks, embedding_model=dummy_model)

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
