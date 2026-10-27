"""Retrieval module for semantic search over indexed document chunks."""
from __future__ import annotations

import logging
from typing import Any

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import AppConfig, load_config
from src.index import (
    get_embedding_model,
    index_exists,
    load_index_and_metadata,
)

logger = logging.getLogger(__name__)


class Retriever:
    """Semantic retriever over a FAISS index with optional reranking."""

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[dict[str, Any]],
        embedding_model: str | SentenceTransformer,
        top_k: int = 5,
        reranker: CrossEncoder | None = None,
        candidate_multiplier: int = 4,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.top_k = top_k
        self.reranker = reranker
        self.candidate_multiplier = max(1, candidate_multiplier)

        if isinstance(embedding_model, str):
            self.model = get_embedding_model(embedding_model)
        else:
            self.model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k chunks, optionally reranking FAISS candidates."""
        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else self.top_k
        k = min(k, len(self.chunks))

        if k <= 0:
            return []

        # Without a reranker, preserve the original FAISS retrieval behavior.
        candidate_k = k
        if self.reranker is not None:
            candidate_k = min(
                k * self.candidate_multiplier,
                len(self.chunks),
            )

        query_vec = self.model.encode(
            [query.strip()],
            normalize_embeddings=True,
        )
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)

        scores, indices = self.index.search(query_vec, candidate_k)

        results: list[dict[str, Any]] = []

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            results.append({
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "doc_id": chunk.get("doc_id", ""),
                "source_file": chunk.get("source_file", ""),
                "title": chunk.get("title", ""),
                "source_url": chunk.get("source_url", ""),
                "pages": chunk.get("pages", []),
                "text": chunk.get("text", ""),
                "score": float(score),
                "chunk_index": chunk.get("chunk_index", 0),
            })

        if self.reranker is not None and results:
            pairs = [
                (query.strip(), chunk["text"])
                for chunk in results
            ]

            rerank_scores = self.reranker.predict(
                pairs,
                show_progress_bar=False,
            )

            for chunk, rerank_score in zip(results, rerank_scores):
                chunk["rerank_score"] = float(rerank_score)

            results.sort(
                key=lambda chunk: chunk["rerank_score"],
                reverse=True,
            )

        return results[:k]


def get_retriever(
    config: AppConfig | None = None,
    index: faiss.Index | None = None,
    chunks: list[dict[str, Any]] | None = None,
    model: SentenceTransformer | None = None,
    reranker: CrossEncoder | None = None,
    candidate_multiplier: int = 4,
) -> Retriever:
    """Create a Retriever, loading the FAISS index when necessary."""
    cfg = config or load_config()

    if index is None or chunks is None:
        if not index_exists(cfg.paths.index):
            raise FileNotFoundError(
                f"FAISS index not found at '{cfg.paths.index}'. "
                "Please build the index first using: python -m src.cli index"
            )

        idx, chs = load_index_and_metadata(cfg.paths.index)
    else:
        idx, chs = index, chunks

    embed_model = model or cfg.models.embedding

    return Retriever(
        index=idx,
        chunks=chs,
        embedding_model=embed_model,
        top_k=cfg.retrieval.top_k,
        reranker=reranker,
        candidate_multiplier=candidate_multiplier,
    )
