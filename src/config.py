"""Configuration loader for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class ModelsConfig:
    generation: str
    embedding: str
    judge: str


@dataclass(frozen=True)
class RetrievalConfig:
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    top_k: int


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    parsed: Path
    index: Path
    manifest: Path


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    refusal: str


@dataclass(frozen=True)
class AppConfig:
    models: ModelsConfig
    retrieval: RetrievalConfig
    paths: PathsConfig
    generation: GenerationConfig
    project_root: Path
    raw_dict: dict[str, Any]


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from config.yaml, resolving relative paths to project root."""
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {target_path}")

    with target_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    root = target_path.resolve().parent

    models_data = data.get("models", {})
    retrieval_data = data.get("retrieval", {})
    paths_data = data.get("paths", {})
    generation_data = data.get("generation", {})

    models_cfg = ModelsConfig(
        generation=str(models_data.get("generation", "qwen2.5:7b")),
        embedding=str(models_data.get("embedding", "BAAI/bge-base-en-v1.5")),
        judge=str(models_data.get("judge", "qwen2.5:7b")),
    )

    retrieval_cfg = RetrievalConfig(
        chunk_size_tokens=int(retrieval_data.get("chunk_size_tokens", 512)),
        chunk_overlap_tokens=int(retrieval_data.get("chunk_overlap_tokens", 64)),
        top_k=int(retrieval_data.get("top_k", 5)),
    )

    def resolve_path(p: str | Path, default: str) -> Path:
        val = Path(p if p else default)
        return val if val.is_absolute() else (root / val).resolve()

    paths_cfg = PathsConfig(
        raw=resolve_path(paths_data.get("raw"), "data/raw"),
        parsed=resolve_path(paths_data.get("parsed"), "data/parsed"),
        index=resolve_path(paths_data.get("index"), "data/index"),
        manifest=resolve_path(paths_data.get("manifest"), "data/manifest.csv"),
    )

    generation_cfg = GenerationConfig(
        temperature=float(generation_data.get("temperature", 0)),
        refusal=str(generation_data.get("refusal", "Not found in the provided documents.")),
    )

    return AppConfig(
        models=models_cfg,
        retrieval=retrieval_cfg,
        paths=paths_cfg,
        generation=generation_cfg,
        project_root=root,
        raw_dict=data,
    )
