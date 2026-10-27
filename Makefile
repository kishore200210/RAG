.PHONY: setup smoke ingest index test lint
setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
smoke:
	pytest -q tests/test_smoke.py
ingest:
	python -m src.ingest
index:
	python -m src.cli index
test:
	pytest -q
lint:
	ruff check .
