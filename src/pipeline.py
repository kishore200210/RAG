"""End-to-end RAG pipeline coordinating retrieval and answer generation."""
from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import CrossEncoder

from src.config import AppConfig, load_config
from src.generation import Generator, get_generator
from src.retrieval import Retriever, get_retriever

logger = logging.getLogger(__name__)

RERANKER_MODEL = "BAAI/bge-reranker-base"


class RAGPipeline:
    """RAG pipeline with optional cross-encoder reranking."""

    def __init__(
        self,
        config: AppConfig | None = None,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        use_reranker: bool = False,
        candidate_multiplier: int = 4,
    ) -> None:
        self.config = config or load_config()

        if retriever is not None:
            self.retriever = retriever
        elif use_reranker:
            reranker = CrossEncoder(RERANKER_MODEL)
            self.retriever = get_retriever(
                self.config,
                reranker=reranker,
                candidate_multiplier=candidate_multiplier,
            )
        else:
            self.retriever = get_retriever(self.config)

        self.generator = generator or get_generator(self.config)
        self.use_reranker = use_reranker

    def answer(
        self,
        question: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant context and generate an answer with citations."""
        clean_q = question.strip()

        if not clean_q:
            return {
                "query": question,
                "answer": self.config.generation.refusal,
                "citations": [],
                "is_refusal": True,
                "retrieved_chunks": [],
            }

        k = (
            top_k
            if top_k is not None
            else self.config.retrieval.top_k
        )

        chunks = self.retriever.retrieve(clean_q, top_k=k)

        gen_result = self.generator.generate(
            query=clean_q,
            retrieved_chunks=chunks,
        )

        return {
            "query": clean_q,
            "answer": gen_result["answer"],
            "citations": gen_result["citations"],
            "is_refusal": gen_result["is_refusal"],
            "retrieved_chunks": chunks,
        }


def format_qa_output(
    result: dict[str, Any],
    include_chunks: bool = False,
) -> str:
    """Format pipeline QA output for display."""
    lines: list[str] = [
        f"Question: {result['query']}",
        f"Answer: {result['answer']}",
    ]

    citations = result.get("citations", [])
    if citations and not result.get("is_refusal"):
        lines.append("\nSources / Citations:")
        for citation in citations:
            lines.append(f"  - {citation}")

    if include_chunks:
        chunks = result.get("retrieved_chunks", [])
        if chunks:
            lines.append(f"\nRetrieved Chunks ({len(chunks)}):")
            for i, chunk in enumerate(chunks, start=1):
                score = chunk.get("rerank_score", chunk.get("score", 0.0))
                lines.append(
                    f"  [{i}] {chunk.get('source_file')} "
                    f"(Pages: {chunk.get('pages')}, Score: {score:.4f})"
                )

    return "\n".join(lines)
