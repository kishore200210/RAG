"""Baseline RAG system package for RBI document QA."""
from src.chunking import chunk_all_documents, chunk_document_text
from src.config import AppConfig, load_config
from src.generation import Generator, get_generator
from src.index import build_faiss_index, build_or_load_index, load_index_and_metadata
from src.pipeline import RAGPipeline, format_qa_output
from src.retrieval import Retriever, get_retriever

__all__ = [
    "AppConfig",
    "Generator",
    "RAGPipeline",
    "Retriever",
    "build_faiss_index",
    "build_or_load_index",
    "chunk_all_documents",
    "chunk_document_text",
    "format_qa_output",
    "get_generator",
    "get_retriever",
    "load_config",
    "load_index_and_metadata",
]
