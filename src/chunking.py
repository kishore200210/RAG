"""Document chunking module for Markdown documents."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

PAGE_PATTERN = re.compile(r"## Page (\d+)")


def load_manifest(manifest_path: str | Path | None) -> dict[str, dict[str, str]]:
    """Load metadata lookup table from manifest.csv indexed by filename and stem."""
    if not manifest_path:
        return {}
    path = Path(manifest_path)
    if not path.is_file():
        return {}

    manifest: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("file", "").strip()
            if filename:
                manifest[filename] = row
                stem = Path(filename).stem
                manifest[stem] = row
    return manifest


def extract_page_markers(text: str) -> list[tuple[int, int]]:
    """Extract (character_offset, page_number) tuples from Markdown text."""
    markers: list[tuple[int, int]] = []
    for match in PAGE_PATTERN.finditer(text):
        try:
            page_num = int(match.group(1))
            markers.append((match.start(), page_num))
        except (ValueError, IndexError):
            continue
    return markers


def resolve_pages(markers: list[tuple[int, int]], start_char: int, end_char: int) -> list[int]:
    """Determine list of 1-based page numbers spanned by [start_char, end_char]."""
    if not markers:
        return [1]

    # Find the active page number at start_char
    start_page = markers[0][1]
    for pos, pnum in markers:
        if pos <= start_char:
            start_page = pnum
        else:
            break

    # Find the active page number at end_char
    end_page = start_page
    for pos, pnum in markers:
        if pos <= end_char:
            end_page = pnum
        else:
            break

    end_page = max(end_page, start_page)
    return list(range(start_page, end_page + 1))


def get_tokenizer(model_name: str) -> Any:
    """Load AutoTokenizer for the given embedding model."""
    return AutoTokenizer.from_pretrained(model_name, model_max_length=10_000_000)


def chunk_document_text(
    text: str,
    tokenizer: Any,
    chunk_size: int,
    chunk_overlap: int,
    doc_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Split a single document's text into fixed token-size chunks with overlap."""
    if not text or not text.strip():
        return []

    if chunk_size <= chunk_overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be strictly greater than chunk_overlap ({chunk_overlap})"
        )

    meta = doc_metadata or {}
    doc_id = str(meta.get("doc_id", "doc"))
    source_file = str(meta.get("source_file", f"{doc_id}.PDF"))
    title = str(meta.get("title", doc_id))
    source_url = str(meta.get("source_url", ""))

    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    total_tokens = len(input_ids)
    if total_tokens == 0:
        return []

    step = chunk_size - chunk_overlap
    slices: list[tuple[int, int]] = []

    if total_tokens <= chunk_size:
        slices.append((0, total_tokens))
    else:
        start = 0
        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            slices.append((start, end))
            if end == total_tokens:
                break
            start += step

    markers = extract_page_markers(text)
    chunks: list[dict[str, Any]] = []

    for chunk_idx, (s_idx, e_idx) in enumerate(slices):
        # Determine character boundaries
        char_start = offsets[s_idx][0]
        char_end = offsets[e_idx - 1][1]

        chunk_text = text[char_start:char_end].strip()
        # Avoid empty chunks
        if not chunk_text:
            continue

        pages = resolve_pages(markers, char_start, char_end)

        chunks.append({
            "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
            "doc_id": doc_id,
            "source_file": source_file,
            "title": title,
            "source_url": source_url,
            "pages": pages,
            "chunk_index": chunk_idx,
            "text": chunk_text,
            "token_count": e_idx - s_idx,
            "char_start": char_start,
            "char_end": char_end,
        })

    return chunks


def chunk_all_documents(
    parsed_dir: str | Path,
    manifest_path: str | Path | None,
    tokenizer: Any,
    chunk_size: int,
    chunk_overlap: int,
    max_docs: int | None = None,
) -> list[dict[str, Any]]:
    """Process all Markdown files in parsed_dir and return a flat list of chunk records."""
    parsed_path = Path(parsed_dir)
    if not parsed_path.exists():
        raise FileNotFoundError(f"Parsed directory not found: {parsed_path}")

    manifest_lookup = load_manifest(manifest_path)
    md_files = sorted(parsed_path.glob("*.md"))
    if not md_files:
        raise ValueError(f"No markdown documents found in {parsed_path}")

    if max_docs is not None and max_docs > 0:
        md_files = md_files[:max_docs]

    all_chunks: list[dict[str, Any]] = []

    for file_path in md_files:
        stem = file_path.stem
        raw_manifest_entry = manifest_lookup.get(stem, manifest_lookup.get(file_path.name, {}))

        doc_meta = {
            "doc_id": stem,
            "source_file": raw_manifest_entry.get("file", f"{stem}.PDF"),
            "title": raw_manifest_entry.get("title", stem),
            "source_url": raw_manifest_entry.get("source_url", ""),
        }

        content = file_path.read_text(encoding="utf-8")
        file_chunks = chunk_document_text(
            text=content,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            doc_metadata=doc_meta,
        )
        all_chunks.extend(file_chunks)

    return all_chunks
