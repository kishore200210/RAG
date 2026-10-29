"""Smoke tests for local LLM and embedding dependencies."""
import os

import pytest


@pytest.mark.skipif(os.getenv("SKIP_MODEL_SMOKE") == "1", reason="Model smoke tests disabled")
def test_ollama_llm():
    ollama = pytest.importorskip("ollama")
    try:
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": "Reply with exactly OK"}],
            options={"temperature": 0},
        )
    except (
        ollama.ResponseError,
        ollama.RequestError,
        ConnectionError,
        TimeoutError,
        OSError,
        RuntimeError,
    ) as exc:
        pytest.fail(f"Ollama/model is not ready. Start Ollama and pull qwen2.5:7b. Error: {exc}")
    assert response["message"]["content"].strip()


@pytest.mark.skipif(os.getenv("SKIP_MODEL_SMOKE") == "1", reason="Model smoke tests disabled")
def test_embedding_model():
    st = pytest.importorskip("sentence_transformers")
    try:
        model = st.SentenceTransformer("BAAI/bge-base-en-v1.5")
        vectors = model.encode(["hello"], normalize_embeddings=True)
    except (
        ConnectionError,
        TimeoutError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as exc:
        pytest.fail(f"Embedding model could not load. Check internet/cache. Error: {exc}")
    assert vectors.shape == (1, 768)
