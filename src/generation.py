"""Answer generation module using Ollama and retrieved document context."""
from __future__ import annotations

import logging
import re
from typing import Any

from src.config import AppConfig, load_config

logger = logging.getLogger(__name__)


def format_context_chunks(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a standardized context block for the prompt."""
    if not chunks:
        return "No relevant context found."

    formatted_blocks: list[str] = []
    for idx, c in enumerate(chunks, start=1):
        source = c.get("source_file", "Unknown")
        pages = c.get("pages", [])
        if pages:
            pages_str = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {pages[0]}-{pages[-1]}"
        else:
            pages_str = "Unknown page"

        header = f"[Context {idx}] Source Document: {source} ({pages_str})"
        text = c.get("text", "").strip()
        formatted_blocks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(formatted_blocks)


def build_rag_prompt(query: str, context: str, refusal_phrase: str) -> str:
    """Construct a prompt for concise, grounded RBI question answering."""
    return f"""You are an expert question-answering assistant for Reserve Bank of India (RBI) documents.

Rules:
1. Answer using ONLY information explicitly supported by the provided context.
2. Give a concise, direct answer. Include relevant figures, dates, conditions, and regulatory terms when applicable.
3. Do not invent facts, speculate, or add unsupported explanations.
4. If the question is ambiguous and cannot be answered without clarification, briefly ask the user to clarify instead of guessing.
5. Strict Refusal Rule: If the context does not contain sufficient information to answer a clear question, return EXACTLY:
{refusal_phrase}
6. When answering from context, cite the supporting source document filename(s), using the filenames shown in the context.
7. Do not claim that a source supports information it does not contain.

Context:
{context}

Question: {query}

Answer:"""


def extract_citations_from_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    """Extract distinct source document citations with page references from chunks."""
    seen: set[str] = set()
    citations: list[str] = []

    for c in chunks:
        source = c.get("source_file", "").strip()
        if not source:
            continue
        pages = c.get("pages", [])
        if pages:
            pages_str = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {pages[0]}-{pages[-1]}"
            label = f"{source} ({pages_str})"
        else:
            label = source

        if label not in seen:
            seen.add(label)
            citations.append(label)

    return citations


def is_refusal_response(response_text: str, refusal_phrase: str) -> bool:
    """Check if the response text represents a refusal."""
    cleaned = response_text.strip().strip("'\"").strip()
    target = refusal_phrase.strip().strip("'\"").strip()

    if cleaned == target:
        return True

    # Check case-insensitive equality or direct prefix/containment
    cleaned_norm = re.sub(r"[^\w\s]", "", cleaned).lower()
    target_norm = re.sub(r"[^\w\s]", "", target).lower()
    return target_norm in cleaned_norm or cleaned_norm == target_norm


class Generator:
    """Local LLM answer generator via Ollama."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        refusal: str = "Not found in the provided documents.",
        ollama_client: Any = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.refusal = refusal
        self._client = ollama_client

    def _call_llm(self, prompt: str) -> str:
        """Execute chat request against Ollama."""
        if self._client is not None:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
        else:
            import ollama

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )

        msg = response.get("message", {})
        return str(msg.get("content", "")).strip()

    def generate(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate answer grounded in retrieved chunks, returning answer and citations."""
        if not retrieved_chunks:
            return {
                "query": query,
                "answer": self.refusal,
                "citations": [],
                "is_refusal": True,
                "raw_response": self.refusal,
            }

        context = format_context_chunks(retrieved_chunks)
        prompt = build_rag_prompt(query=query, context=context, refusal_phrase=self.refusal)

        raw_response = self._call_llm(prompt)

        if is_refusal_response(raw_response, self.refusal):
            return {
                "query": query,
                "answer": self.refusal,
                "citations": [],
                "is_refusal": True,
                "raw_response": raw_response,
            }

        citations = extract_citations_from_chunks(retrieved_chunks)

        return {
            "query": query,
            "answer": raw_response,
            "citations": citations,
            "is_refusal": False,
            "raw_response": raw_response,
        }


def get_generator(config: AppConfig | None = None, ollama_client: Any = None) -> Generator:
    """Create a Generator using settings from config."""
    cfg = config or load_config()
    return Generator(
        model=cfg.models.generation,
        temperature=cfg.generation.temperature,
        refusal=cfg.generation.refusal,
        ollama_client=ollama_client,
    )
