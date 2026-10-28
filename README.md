# Open-Source RAG System with Evaluation Harness

Python 3.11 project for question answering over RBI documents. Models run locally through Ollama; no paid LLM APIs are used.

## Quick start (Ubuntu/Linux)

```bash
python3.11 --version
ollama --version
ollama pull qwen2.5:7b
ollama run qwen2.5:7b "Say hello"
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pytest -q tests/test_smoke.py
```

## Corpus

Run `python scripts/download_corpus.py`. The scraper uses a polite delay and records downloaded documents in `data/manifest.csv`. If scraping is blocked, manually download RBI PDFs and add their metadata to the manifest. Keep raw and parsed PDFs out of Git.

Then run `python -m src.ingest` to parse PDFs into Markdown files.

## Baseline RAG Pipeline

The baseline RAG pipeline reads parsed Markdown files, chunks them into 512 tokens with 64-token overlap, indexes them using FAISS and `BAAI/bge-base-en-v1.5`, retrieves top-k chunks, and generates grounded answers with source citations using local Ollama `qwen2.5:7b`.

### 1. Build or Rebuild the Index

Build FAISS index over all documents in `data/parsed/`:

```bash
python -m src.cli index
# or via Makefile
make index
# or force a rebuild
python -m src.cli index --rebuild
```

### 2. Ask Questions

Query the pipeline from the command line:

```bash
python -m src.cli ask "What regulations govern borrowing and lending in Indian Rupees between residents and non-residents?"
# With retrieved chunk details:
python -m src.cli ask "What regulations govern borrowing and lending in Indian Rupees between residents and non-residents?" --show-chunks
```

If the context does not contain the answer, the system strictly outputs:
`Not found in the provided documents.`

### 3. Run Tests

Run all unit, integration, and smoke tests:

```bash
pytest -q
# or
make test
```
