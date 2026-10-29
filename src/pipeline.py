"""End-to-end RAG pipeline coordinating retrieval and answer generation."""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

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
            try:
                from sentence_transformers import CrossEncoder

                reranker = CrossEncoder(RERANKER_MODEL, local_files_only=True)
            except (OSError, FileNotFoundError, ValueError):
                from sentence_transformers import CrossEncoder

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
        t0_total = time.perf_counter()
        clean_q = question.strip()

        if not clean_q:
            return {
                "query": question,
                "answer": self.config.generation.refusal,
                "citations": [],
                "is_refusal": True,
                "retrieved_chunks": [],
                "timings": {
                    "embedding_model_init_seconds": round(getattr(self.retriever, "embedding_init_time", 0.0), 4),
                    "query_embedding_seconds": 0.0,
                    "faiss_retrieval_seconds": 0.0,
                    "reranking_seconds": 0.0,
                    "prompt_construction_seconds": 0.0,
                    "llm_generation_seconds": 0.0,
                    "total_execution_seconds": round(time.perf_counter() - t0_total, 4),
                },
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

        t_total = time.perf_counter() - t0_total

        ret_timings = getattr(self.retriever, "last_timings", {})
        if not isinstance(ret_timings, dict):
            ret_timings = {}
        gen_timings = getattr(self.generator, "last_timings", {})
        if not isinstance(gen_timings, dict):
            gen_timings = {}

        timings = {
            "embedding_model_init_seconds": round(getattr(self.retriever, "embedding_init_time", 0.0), 4),
            "query_embedding_seconds": round(ret_timings.get("query_embedding_seconds", 0.0), 4),
            "faiss_retrieval_seconds": round(ret_timings.get("faiss_retrieval_seconds", 0.0), 4),
            "reranking_seconds": round(ret_timings.get("reranking_seconds", 0.0), 4),
            "prompt_construction_seconds": round(gen_timings.get("prompt_construction_seconds", 0.0), 6),
            "llm_generation_seconds": round(gen_timings.get("llm_generation_seconds", 0.0), 4),
            "total_execution_seconds": round(t_total, 4),
        }

        logger.info("Pipeline stage timings: %s", timings)

        return {
            "query": clean_q,
            "answer": gen_result["answer"],
            "citations": gen_result["citations"],
            "is_refusal": gen_result["is_refusal"],
            "retrieved_chunks": chunks,
            "context_chunks": gen_result.get("context_chunks", chunks),
            "timings": timings,
        }

    def answer_stream(
        self,
        question: str,
        top_k: int | None = None,
    ) -> tuple[Iterator[str], list[dict[str, Any]]]:
        """Retrieve relevant context and stream answer tokens, returning generator and chunks."""
        clean_q = question.strip()

        if not clean_q:
            def _refusal_iter() -> Iterator[str]:
                yield self.config.generation.refusal

            return _refusal_iter(), []

        k = (
            top_k
            if top_k is not None
            else self.config.retrieval.top_k
        )

        chunks = self.retriever.retrieve(clean_q, top_k=k)
        stream = self.generator.generate_stream(
            query=clean_q,
            retrieved_chunks=chunks,
        )
        return stream, chunks


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
