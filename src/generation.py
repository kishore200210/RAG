"""Answer generation module using Ollama and retrieved document context."""
from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterator
from typing import Any

from src.config import AppConfig, load_config

logger = logging.getLogger(__name__)


def estimate_token_count(text: str) -> int:
    """Fast estimation of BPE tokens for text (roughly ~1.3 tokens per word or 4 chars/token)."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(1, max(chars // 4, int(words * 1.3)))


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens, breaking cleanly on a sentence or clause boundary."""
    words = text.split()
    target_words = max(1, int(max_tokens / 1.3))
    if len(words) <= target_words:
        return text

    truncated_words = words[:target_words]
    candidate = " ".join(truncated_words)
    last_punct = max(
        candidate.rfind(". "),
        candidate.rfind(".\n"),
        candidate.rfind("! "),
        candidate.rfind("? "),
        candidate.rfind(": "),
        candidate.rfind(":\n"),
        candidate.rfind(";\n"),
    )
    if last_punct > len(candidate) // 2:
        return candidate[:last_punct + 1]
    return candidate


def select_context_chunks(
    chunks: list[dict[str, Any]],
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Select and prioritize chunks within the token budget.

    Candidate chunks are prioritized by their incoming rank (relevance score).
    Chunks that fit within max_tokens are preserved in full. If the next chunk
    exceeds the remaining budget but there is meaningful capacity (>= 120 tokens),
    it is cleanly truncated at sentence boundaries. Lower-priority chunks exceeding
    the budget are omitted, reducing prompt-prefill latency while keeping relevant evidence.
    """
    if not chunks:
        return []
    if max_tokens is None or max_tokens <= 0:
        return list(chunks)

    selected: list[dict[str, Any]] = []
    total_tokens = 0

    for idx, c in enumerate(chunks, start=1):
        text = c.get("text", "").strip()
        source = c.get("source_file", "Unknown")
        pages = c.get("pages", [])
        if pages:
            pages_str = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {pages[0]}-{pages[-1]}"
        else:
            pages_str = "Unknown page"
        header = f"[Context {idx}] Source Document: {source} ({pages_str})"
        header_tokens = estimate_token_count(header) + 4
        chunk_tokens = estimate_token_count(text)
        block_tokens = header_tokens + chunk_tokens

        if total_tokens + block_tokens <= max_tokens:
            selected.append(c)
            total_tokens += block_tokens
        else:
            remaining = max_tokens - total_tokens - header_tokens
            if not selected:
                if remaining > 100:
                    truncated_text = _truncate_text_to_tokens(text, remaining)
                    truncated_c = dict(c)
                    truncated_c["text"] = truncated_text
                    selected.append(truncated_c)
                else:
                    selected.append(c)
                break
            elif remaining >= 120:
                truncated_text = _truncate_text_to_tokens(text, remaining)
                if truncated_text:
                    truncated_c = dict(c)
                    truncated_c["text"] = truncated_text
                    selected.append(truncated_c)
                break
            else:
                break

    logger.debug(
        "Context budgeting: selected %d/%d chunks (~%d est tokens, budget %d)",
        len(selected),
        len(chunks),
        total_tokens,
        max_tokens,
    )
    return selected


def format_context_chunks(
    chunks: list[dict[str, Any]],
    max_tokens: int | None = None,
) -> str:
    """Format retrieved chunks into a standardized context block for the prompt.

    If max_tokens is specified, chunks are prioritized by relevance within the token budget.
    """
    if not chunks:
        return "No relevant context found."

    selected = select_context_chunks(chunks, max_tokens=max_tokens) if max_tokens else chunks
    if not selected:
        return "No relevant context found."

    formatted_blocks: list[str] = []
    for idx, c in enumerate(selected, start=1):
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
1. Answer using only the provided context. Never invent or infer unsupported facts.
2. For specific figures, dates, rates, fees, penalties, limits, or requirements, verify explicit supporting evidence.
3. Strict Refusal Rule: If evidence is insufficient, return exactly: {refusal_phrase}
4. If the question is ambiguous, briefly ask for clarification.
5. Cite supporting source filenames when answering. Do not add citations when refusing.
6. Give a direct answer in no more than two short sentences. Preserve important facts, conditions, and regulatory terms.
7. When refusing, output only the exact refusal phrase.

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
    """Check whether the response explicitly declines to answer due to missing evidence."""
    cleaned = response_text.strip().strip("'\"").strip()
    target = refusal_phrase.strip().strip("'\"").strip()

    cleaned_norm = re.sub(r"[^\w\s]", "", cleaned).lower()
    target_norm = re.sub(r"[^\w\s]", "", target).lower()

    # Exact refusal phrase.
    if cleaned_norm == target_norm:
        return True

    # Recognize explicit statements that the requested information is absent.
    missing_evidence_patterns = [
        r"\bnot provided in (?:the )?(?:given )?contexts?\b",
        r"\bdo not contain any information\b",
        r"\bdoes not contain any information\b",
        r"\bno information (?:is )?(?:provided|available|given)\b",
        r"\bnot mentioned in (?:the )?(?:provided )?(?:documents?|context|text)\b",
        r"\bnot (?:possible|able) to determine\b",
        r"\bcannot determine\b",
        r"\bcannot be determined\b",
        r"\bthere is no (?:mention|information|data)\b",
        # Additional variants observed in practice from qwen2.5:7b outputs
        r"\bdoes not contain (?:any )?(?:data|details?|information)\b",
        r"\bdo not contain (?:any )?(?:data|details?|information)\b",
        r"\bnot (?:found|available|present) in (?:the )?(?:given |provided )?(?:documents?|contexts?|text)\b",
        r"\bthere is no specific (?:mention|information|data|detail)\b",
        r"\bdoes not (?:have|include|contain) (?:the )?(?:specific )?(?:information|data|details?)\b",
    ]

    return any(
        re.search(pattern, cleaned, flags=re.IGNORECASE)
        for pattern in missing_evidence_patterns
    )

def _get_optimal_threads() -> int:
    """Determine optimal thread count (capped at 4) to prevent CPU cache thrashing on CPU inference."""
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical and physical > 0:
            return min(4, int(physical))
    except (ImportError, AttributeError, OSError) as exc:
        logger.debug("Failed to detect physical CPU count with psutil: %s", exc)

    count = os.cpu_count() or 4
    return min(4, max(1, count // 2 if count > 4 else count))



class Generator:
    """Local LLM answer generator via Ollama."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        temperature: float = 0.0,
        refusal: str = "Not found in the provided documents.",
        context_token_budget: int = 400,
        ollama_client: Any = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.refusal = refusal
        self.context_token_budget = context_token_budget
        self._client = ollama_client
        self.last_timings: dict[str, float] = {}

    def _call_llm(self, prompt: str) -> str:
        """Execute chat request against Ollama."""
        options = {
            "temperature": self.temperature,
            "num_predict": 100,
            "num_thread": _get_optimal_threads(),
        }
        if self._client is not None:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=options,
                keep_alive="1h",
            )
        else:
            import ollama

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=options,
                keep_alive="1h",
            )

        msg = response.get("message", {})
        return str(msg.get("content", "")).strip()

    def generate_stream(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> Iterator[str]:
        """Stream answer tokens grounded in retrieved chunks."""
        if not retrieved_chunks:
            yield self.refusal
            return

        selected_chunks = select_context_chunks(
            retrieved_chunks,
            max_tokens=self.context_token_budget,
        )
        context = format_context_chunks(selected_chunks)
        prompt = build_rag_prompt(query=query, context=context, refusal_phrase=self.refusal)
        options = {
            "temperature": self.temperature,
            "num_predict": 100,
            "num_thread": _get_optimal_threads(),
        }

        if self._client is not None:
            stream = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=options,
                keep_alive="1h",
                stream=True,
            )
        else:
            import ollama

            stream = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=options,
                keep_alive="1h",
                stream=True,
            )

        for chunk in stream:
            msg = chunk.get("message", {})
            content = msg.get("content", "")
            if content:
                yield str(content)

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
                "context_chunks": [],
            }

        t0_prompt = time.perf_counter()
        selected_chunks = select_context_chunks(
            retrieved_chunks,
            max_tokens=self.context_token_budget,
        )
        context = format_context_chunks(selected_chunks)
        prompt = build_rag_prompt(query=query, context=context, refusal_phrase=self.refusal)
        t_prompt = time.perf_counter() - t0_prompt

        t0_llm = time.perf_counter()
        raw_response = self._call_llm(prompt)
        t_llm = time.perf_counter() - t0_llm

        self.last_timings = {
            "prompt_construction_seconds": t_prompt,
            "llm_generation_seconds": t_llm,
            "prompt_estimated_tokens": float(estimate_token_count(prompt)),
            "context_chunks_count": float(len(selected_chunks)),
        }

        if is_refusal_response(raw_response, self.refusal):
            return {
                "query": query,
                "answer": self.refusal,
                "citations": [],
                "is_refusal": True,
                "raw_response": raw_response,
                "context_chunks": selected_chunks,
            }

        citations = extract_citations_from_chunks(selected_chunks)

        return {
            "query": query,
            "answer": raw_response,
            "citations": citations,
            "is_refusal": False,
            "raw_response": raw_response,
            "context_chunks": selected_chunks,
        }


def get_generator(config: AppConfig | None = None, ollama_client: Any = None) -> Generator:
    """Create a Generator using settings from config."""
    cfg = config or load_config()
    return Generator(
        model=cfg.models.generation,
        temperature=cfg.generation.temperature,
        refusal=cfg.generation.refusal,
        context_token_budget=cfg.generation.context_token_budget,
        ollama_client=ollama_client,
    )

