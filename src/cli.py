"""Command-line interface for the RBI RAG pipeline."""
from __future__ import annotations

import argparse
import logging
import sys

from src.config import load_config
from src.index import build_or_load_index
from src.pipeline import RAGPipeline, format_qa_output

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-cli",
        description="RBI Document Question-Answering Baseline RAG CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: index
    index_parser = subparsers.add_parser("index", help="Build or rebuild the FAISS vector index")
    index_parser.add_argument("--rebuild", action="store_true", help="Force rebuild the index from scratch")
    index_parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    index_parser.add_argument("--max-docs", type=int, default=None, help="Maximum number of documents to index (default: all)")

    # Command: ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question over the indexed RBI documents")
    ask_parser.add_argument("question", nargs="?", help="Question text (prompted interactively if omitted)")
    ask_parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve (default from config)")
    ask_parser.add_argument("--show-chunks", action="store_true", help="Display retrieved chunk details and scores")
    ask_parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")

    # Direct top-level flags as convenient aliases
    parser.add_argument("--index", action="store_true", help="Alias for 'index' command")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild when indexing")
    parser.add_argument("--query", "-q", type=str, help="Alias for 'ask' command with query text")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum number of documents to index")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")


    return parser


def handle_index(rebuild: bool, batch_size: int = 64, max_docs: int | None = None) -> None:
    cfg = load_config()
    print(f"Building/verifying index at {cfg.paths.index} from {cfg.paths.parsed}...")
    index, chunks = build_or_load_index(
        cfg,
        force_rebuild=rebuild,
        batch_size=batch_size,
        max_docs=max_docs,
    )
    print(f"Index successfully prepared with {index.ntotal} vectors across {len(chunks)} chunks.")


def handle_ask(
    question: str,
    top_k: int | None = None,
    show_chunks: bool = False,
    use_reranker: bool = False,
) -> None:
    cfg = load_config()
    try:
        pipeline = RAGPipeline(config=cfg, use_reranker=use_reranker)
    except FileNotFoundError as err:
        print(f"\n[Error] {err}\n", file=sys.stderr)
        sys.exit(1)

    result = pipeline.answer(question=question, top_k=top_k)
    output = format_qa_output(result, include_chunks=show_chunks)
    print("\n" + output)

    if "timings" in result:
        t = result["timings"]
        print(
            f"\n[Latency Breakdown] Model Init: {t.get('embedding_model_init_seconds', 0):.3f}s | "
            f"Embed: {t.get('query_embedding_seconds', 0):.3f}s | "
            f"FAISS: {t.get('faiss_retrieval_seconds', 0):.3f}s | "
            f"LLM: {t.get('llm_generation_seconds', 0):.3f}s | "
            f"Total Answer: {t.get('total_execution_seconds', 0):.3f}s\n"
        )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    use_reranker = getattr(args, "rerank", False)

    if args.index:
        handle_index(rebuild=args.rebuild, max_docs=args.max_docs)
        return

    if args.query:
        handle_ask(question=args.query, top_k=getattr(args, "top_k", None), use_reranker=use_reranker)
        return

    if args.command == "index":
        handle_index(rebuild=args.rebuild, batch_size=args.batch_size, max_docs=args.max_docs)
    elif args.command == "ask":
        question = args.question
        if not question:
            try:
                question = input("Enter your question: ").strip()
            except (KeyboardInterrupt, EOFError):
                sys.exit(0)
        if not question:
            print("No question provided.")
            sys.exit(1)
        handle_ask(
            question=question,
            top_k=args.top_k,
            show_chunks=args.show_chunks,
            use_reranker=use_reranker,
        )
    else:
        # If user passed a positional question without 'ask'
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            handle_ask(question=" ".join(sys.argv[1:]), use_reranker=use_reranker)
        else:
            parser.print_help()



if __name__ == "__main__":
    main()
