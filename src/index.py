"""Embedding generation and FAISS index persistence module."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import chunk_all_documents, get_tokenizer
from src.config import AppConfig, load_config

logger = logging.getLogger(__name__)

FAISS_INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "chunks.json"


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load SentenceTransformer embedding model."""
    return SentenceTransformer(model_name)


def index_exists(index_dir: str | Path) -> bool:
    """Check whether both the FAISS index and chunk metadata files exist."""
    dir_path = Path(index_dir)
    return (dir_path / FAISS_INDEX_FILENAME).is_file() and (dir_path / METADATA_FILENAME).is_file()


def save_index_and_metadata(
    index: faiss.Index,
    chunks: list[dict[str, Any]],
    index_dir: str | Path,
) -> None:
    """Persist FAISS index and chunks metadata to disk."""
    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = out_dir / FAISS_INDEX_FILENAME
    metadata_path = out_dir / METADATA_FILENAME

    faiss.write_index(index, str(faiss_path))
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info("Saved FAISS index to %s and metadata to %s", faiss_path, metadata_path)


def load_index_and_metadata(
    index_dir: str | Path,
) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Load previously saved FAISS index and chunks metadata from disk."""
    dir_path = Path(index_dir)
    faiss_path = dir_path / FAISS_INDEX_FILENAME
    metadata_path = dir_path / METADATA_FILENAME

    if not faiss_path.is_file():
        raise FileNotFoundError(f"FAISS index file not found at {faiss_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    index = faiss.read_index(str(faiss_path))
    with metadata_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info("Loaded %d vectors from %s", index.ntotal, faiss_path)
    return index, chunks


def build_faiss_index(
    chunks: list[dict[str, Any]],
    model_name_or_instance: str | SentenceTransformer,
    index_dir: str | Path | None = None,
    batch_size: int = 64,
    show_progress_bar: bool = True,
) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Compute normalized embeddings and construct a FAISS cosine similarity index."""
    if not chunks:
        raise ValueError("Cannot build index from empty chunk list")

    if isinstance(model_name_or_instance, str):
        model = get_embedding_model(model_name_or_instance)
    else:
        model = model_name_or_instance

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    if index_dir:
        save_index_and_metadata(index, chunks, index_dir)

    return index, chunks


def build_or_load_index(
    config: AppConfig | None = None,
    force_rebuild: bool = False,
    batch_size: int = 64,
    show_progress_bar: bool = True,
    max_docs: int | None = None,
) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Load existing index or build a new one from parsed documents."""
    cfg = config or load_config()
    index_dir = cfg.paths.index

    if not force_rebuild and index_exists(index_dir):
        logger.info("Loading existing index from %s", index_dir)
        return load_index_and_metadata(index_dir)

    logger.info("Building new FAISS index from %s", cfg.paths.parsed)
    tokenizer = get_tokenizer(cfg.models.embedding)
    chunks = chunk_all_documents(
        parsed_dir=cfg.paths.parsed,
        manifest_path=cfg.paths.manifest,
        tokenizer=tokenizer,
        chunk_size=cfg.retrieval.chunk_size_tokens,
        chunk_overlap=cfg.retrieval.chunk_overlap_tokens,
        max_docs=max_docs,
    )

    return build_faiss_index(
        chunks=chunks,
        model_name_or_instance=cfg.models.embedding,
        index_dir=index_dir,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or rebuild the FAISS index.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the index even if it exists")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum number of documents to index (default: all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    index, chunks = build_or_load_index(
        cfg,
        force_rebuild=args.rebuild,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
    )
    print(f"Index ready: {index.ntotal} vectors indexed from {len(chunks)} chunks in {cfg.paths.index}")


if __name__ == "__main__":
    main()
