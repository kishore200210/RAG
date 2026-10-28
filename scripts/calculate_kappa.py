#!/usr/bin/env python3
"""
Calculate Cohen's Kappa between Human Reference Labels and LLM Judge Predictions.

Usage:
    python scripts/calculate_kappa.py [path_to_labeled_csv]

Defaults to: eval/human_validation_20_template.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score, confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Cohen's kappa between human labels and judge predictions."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="eval/human_validation_20_template.csv",
        type=Path,
        help="Path to CSV containing human labels and judge predictions.",
    )
    return parser.parse_args()


def normalize_correctness(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in {"", "none", "nan", "n/a", "na", "skip", "skipped"}:
        return None
    try:
        f = float(s)
        if f in {0.0, 0.5, 1.0}:
            return f
    except ValueError:
        pass
    if s in {"correct", "true", "yes", "1"}:
        return 1.0
    if s in {"partial", "partially", "partially_correct", "0.5"}:
        return 0.5
    if s in {"incorrect", "false", "no", "0"}:
        return 0.0
    return None


def normalize_answerability(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in {"", "none", "nan"}:
        return None
    if s in {"1", "true", "answerable", "yes"}:
        return "answerable"
    if s in {"0", "false", "unanswerable", "no"}:
        return "unanswerable"
    if s in {"ambiguous", "ambig", "unclear"}:
        return "ambiguous"
    return s


def parse_judge_correctness(judge_pred_str: str) -> float | None:
    """Extract numeric correctness score from Judge prediction string."""
    s = str(judge_pred_str).strip()
    if "SKIPPED" in s.upper():
        return None
    # Look for correctness=X
    import re
    m = re.search(r"correctness=([0-9.]+)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def main() -> int:
    args = parse_args()
    csv_path: Path = args.csv_path

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} samples from: {csv_path}")
    print("-" * 60)

    # Check how many human labels are populated
    labeled_correctness = []
    labeled_answerability = []

    for r in rows:
        h_corr_raw = r.get("Human correctness label, where applicable", "")
        h_ans_raw = r.get("Human answerability label", "")
        
        h_corr = normalize_correctness(h_corr_raw)
        h_ans = normalize_answerability(h_ans_raw)
        
        j_corr = parse_judge_correctness(r.get("Judge prediction", ""))
        
        qid = r.get("Question ID", "")
        
        if h_corr is not None and j_corr is not None:
            labeled_correctness.append((qid, h_corr, j_corr))
            
        if h_ans is not None:
            labeled_answerability.append((qid, h_ans, r.get("Judge prediction", "")))

    print(f"Human answerability labels provided: {len(labeled_answerability)} / {len(rows)}")
    print(f"Human correctness labels provided (for judged samples): {len(labeled_correctness)} / {len(rows)}")

    if not labeled_correctness and not labeled_answerability:
        print("\n[STATUS: PENDING HUMAN LABELS]")
        print("Cohen's kappa CANNOT be calculated because human reference labels are blank.")
        print("\nWorkflow to complete human evaluation:")
        print("  1. Open eval/human_validation_20_template.csv")
        print("  2. In 'Human answerability label', enter:")
        print("       'answerable', 'unanswerable', or 'ambiguous'")
        print("  3. In 'Human correctness label, where applicable', enter:")
        print("       1.0 (correct / valid refusal), 0.5 (partially correct), or 0.0 (incorrect)")
        print("       (Leave ambiguous questions blank or N/A)")
        print("  4. Save the file and re-run: python scripts/calculate_kappa.py")
        return 0

    # Calculate Correctness Kappa if labels exist
    if labeled_correctness:
        print("\n" + "=" * 60)
        print("1. CORRECTNESS AGREEMENT (Human vs. LLM Judge)")
        print("=" * 60)
        y_human = [item[1] for item in labeled_correctness]
        y_judge = [item[2] for item in labeled_correctness]
        
        labels_sorted = sorted(set(y_human + y_judge))
        
        kappa_unweighted = cohen_kappa_score(y_human, y_judge)
        kappa_linear = cohen_kappa_score(y_human, y_judge, weights="linear")
        kappa_quadratic = cohen_kappa_score(y_human, y_judge, weights="quadratic")
        
        cm = confusion_matrix(y_human, y_judge, labels=labels_sorted)
        
        print(f"Evaluated pairs (n): {len(y_human)}")
        print(f"Categories: {labels_sorted}")
        print("\nConfusion Matrix (Rows = Human, Cols = LLM Judge):")
        header = "Human \\ Judge\t" + "\t".join(str(l) for l in labels_sorted)
        print(header)
        for i, row in enumerate(cm):
            print(f"{labels_sorted[i]}\t\t" + "\t".join(str(c) for c in row))
            
        print("\nCohen's Kappa Statistics:")
        print(f"  Unweighted Kappa (nominal):  {kappa_unweighted:.4f}")
        print(f"  Linear Weighted Kappa:       {kappa_linear:.4f}")
        print(f"  Quadratic Weighted Kappa:    {kappa_quadratic:.4f}")
        
        # Qualitative interpretation (Landis & Koch, 1977)
        def interpret_kappa(k: float) -> str:
            if k < 0:
                return "Poor / Systematic Disagreement (< 0.0)"
            elif k <= 0.20:
                return "Slight agreement (0.01 - 0.20)"
            elif k <= 0.40:
                return "Fair agreement (0.21 - 0.40)"
            elif k <= 0.60:
                return "Moderate agreement (0.41 - 0.60)"
            elif k <= 0.80:
                return "Substantial agreement (0.61 - 0.80)"
            else:
                return "Almost perfect agreement (0.81 - 1.00)"
                
        print(f"  Interpretation:              {interpret_kappa(kappa_unweighted)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
