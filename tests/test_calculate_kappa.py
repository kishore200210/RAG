"""Unit tests for Cohen's kappa calculation script."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from scripts.calculate_kappa import (
    normalize_answerability,
    normalize_correctness,
    parse_judge_correctness,
)


def test_normalize_correctness():
    assert normalize_correctness("1") == 1.0
    assert normalize_correctness("1.0") == 1.0
    assert normalize_correctness("0.5") == 0.5
    assert normalize_correctness("0") == 0.0
    assert normalize_correctness("correct") == 1.0
    assert normalize_correctness("partial") == 0.5
    assert normalize_correctness("incorrect") == 0.0
    assert normalize_correctness("") is None
    assert normalize_correctness("N/A") is None
    assert normalize_correctness("skipped") is None


def test_normalize_answerability():
    assert normalize_answerability("answerable") == "answerable"
    assert normalize_answerability("1") == "answerable"
    assert normalize_answerability("true") == "answerable"
    assert normalize_answerability("unanswerable") == "unanswerable"
    assert normalize_answerability("0") == "unanswerable"
    assert normalize_answerability("false") == "unanswerable"
    assert normalize_answerability("ambiguous") == "ambiguous"
    assert normalize_answerability("") is None


def test_parse_judge_correctness():
    assert parse_judge_correctness("correctness=1.0, faithfulness=1.0 (answer)") == 1.0
    assert parse_judge_correctness("correctness=0.5, faithfulness=0.8 (answer)") == 0.5
    assert parse_judge_correctness("correctness=0.0, faithfulness=1.0 (refusal)") == 0.0
    assert parse_judge_correctness("SKIPPED (ambiguous / no gold answer)") is None
    assert parse_judge_correctness("invalid") is None


def test_kappa_calculation_with_dummy_csv(tmp_path: Path, monkeypatch):
    test_csv = tmp_path / "test_kappa.csv"
    rows = [
        {
            "Question ID": "Q01",
            "Question": "Question 1",
            "Human answerability label": "answerable",
            "Human correctness label, where applicable": "1.0",
            "Judge prediction": "correctness=1.0, faithfulness=1.0",
            "Notes": "",
        },
        {
            "Question ID": "Q02",
            "Question": "Question 2",
            "Human answerability label": "answerable",
            "Human correctness label, where applicable": "0.0",
            "Judge prediction": "correctness=0.0, faithfulness=0.0",
            "Notes": "",
        },
        {
            "Question ID": "Q03",
            "Question": "Question 3",
            "Human answerability label": "unanswerable",
            "Human correctness label, where applicable": "1.0",
            "Judge prediction": "correctness=1.0, faithfulness=1.0",
            "Notes": "",
        },
        {
            "Question ID": "Q04",
            "Question": "Question 4",
            "Human answerability label": "ambiguous",
            "Human correctness label, where applicable": "",
            "Judge prediction": "SKIPPED",
            "Notes": "",
        },
    ]

    with test_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    from scripts.calculate_kappa import main

    monkeypatch.setattr(sys, "argv", ["calculate_kappa.py", str(test_csv)])
    assert main() == 0
