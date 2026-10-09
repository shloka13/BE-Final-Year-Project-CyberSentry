"""Reproducible evaluation script for the CyberSentry Secondary Review Heuristic.

Evaluates the secondary-review heuristic rules defined in
``src.agents.investigation.SecondaryReviewRule`` against labeled validation
and held-out test datasets.

Usage:
    python -m src.ml.evaluate_secondary_review
    python -m src.ml.evaluate_secondary_review --split test
    python -m src.ml.evaluate_secondary_review --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.investigation import SecondaryReviewRule
from src.ml.common import REPORTS, load_table


def evaluate_split(
    df: pd.DataFrame,
    split_name: str,
    rule: SecondaryReviewRule,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a secondary review rule against a labeled dataset split."""
    total_flows = len(df)
    labels = df["label"].astype(str)

    # Required features
    required_features = ["Subflow Fwd Bytes", "Bwd Packet Length Min", "Max Packet Length"]
    if rule.target_ports is not None:
        required_features.append("Destination Port")

    # Check for missing or unusable features
    missing_counts = {}
    for feat in required_features:
        if feat not in df.columns:
            missing_counts[feat] = total_flows
        else:
            # Count null, NaN, or non-finite values
            col = df[feat]
            nulls = int(col.isna().sum())
            non_finite = int((~np.isfinite(col.dropna())).sum()) if len(col) > 0 else 0
            missing_counts[feat] = nulls + non_finite

    # Vectorized evaluation matching SecondaryReviewRule.evaluate() semantics
    fwd = df["Subflow Fwd Bytes"].astype(float)
    bwd_min = df["Bwd Packet Length Min"].astype(float)
    max_pkt = df["Max Packet Length"].astype(float)

    cond = (
        (fwd >= 0.0)
        & (fwd <= rule.max_fwd_bytes)
        & (bwd_min >= 0.0)
        & (bwd_min <= rule.max_bwd_packet_min)
        & (max_pkt >= 0.0)
        & (max_pkt <= rule.max_packet_length)
    )

    if rule.target_ports is not None and "Destination Port" in df.columns:
        dest_port = df["Destination Port"].astype(float).astype(int)
        cond = cond & dest_port.isin(rule.target_ports)

    flagged_mask = cond.fillna(False).to_numpy()
    total_flagged = int(flagged_mask.sum())
    review_rate = float(total_flagged / total_flows) if total_flows > 0 else 0.0

    # Class-level metrics
    per_class_rows: list[dict[str, Any]] = []
    classes = sorted(labels.unique())

    botnet_flagged = 0
    botnet_total = 0
    botnet_rate = 0.0

    benign_flagged = 0
    benign_total = 0
    benign_fpr = 0.0

    for cls in classes:
        cls_mask = (labels == cls).to_numpy()
        cls_total = int(cls_mask.sum())
        cls_flagged = int((flagged_mask & cls_mask).sum())
        cls_rate = float(cls_flagged / cls_total) if cls_total > 0 else 0.0

        if cls == "Botnet":
            botnet_flagged = cls_flagged
            botnet_total = cls_total
            botnet_rate = cls_rate
        elif cls == "Benign":
            benign_flagged = cls_flagged
            benign_total = cls_total
            benign_fpr = cls_rate

        per_class_rows.append({
            "split": split_name,
            "class": cls,
            "total_flows": cls_total,
            "flagged_flows": cls_flagged,
            "flagging_rate": round(cls_rate, 6),
            "percentage": f"{cls_rate:.2%}",
        })

    summary = {
        "split": split_name,
        "total_flows": total_flows,
        "total_flagged_flows": total_flagged,
        "review_rate": round(review_rate, 6),
        "review_rate_pct": f"{review_rate:.2%}",
        "botnet_flows_flagged": botnet_flagged,
        "botnet_flows_total": botnet_total,
        "botnet_flagging_rate": round(botnet_rate, 6),
        "botnet_flagging_rate_pct": f"{botnet_rate:.2%}",
        "benign_flows_flagged": benign_flagged,
        "benign_flows_total": benign_total,
        "benign_false_positive_rate": round(benign_fpr, 6),
        "benign_fpr_pct": f"{benign_fpr:.4%}",
        "missing_or_unusable_features": missing_counts,
        "rule_config": {
            "name": rule.name,
            "max_fwd_bytes": rule.max_fwd_bytes,
            "max_bwd_packet_min": rule.max_bwd_packet_min,
            "max_packet_length": rule.max_packet_length,
            "target_ports": list(rule.target_ports) if rule.target_ports else None,
        },
    }

    return summary, per_class_rows


def run_evaluation(
    splits: list[str] | None = None,
    save_artifacts: bool = True,
    target_ports: list[int] | None = None,
) -> dict[str, Any]:
    """Execute evaluation and return structured results."""
    if splits is None:
        splits = ["val", "test"]

    rule = SecondaryReviewRule(target_ports=target_ports)

    all_summaries: dict[str, Any] = {}
    all_per_class: list[dict[str, Any]] = []

    for s in splits:
        df = load_table(s)
        summary, per_class = evaluate_split(df, s, rule)
        all_summaries[s] = summary
        all_per_class.extend(per_class)

    if save_artifacts:
        REPORTS.mkdir(parents=True, exist_ok=True)
        # 1. Machine-readable JSON summary
        json_path = REPORTS / "secondary_review_metrics.json"
        json_path.write_text(json.dumps(all_summaries, indent=2))

        # 2. Machine-readable CSV per-class breakdown
        csv_path = REPORTS / "secondary_review_per_class.csv"
        pd.DataFrame(all_per_class).to_csv(csv_path, index=False)

    return all_summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the CyberSentry Secondary Review heuristic on validation and test splits."
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "both"],
        default="both",
        help="Dataset split to evaluate (default: both).",
    )
    parser.add_argument(
        "--port-filter",
        type=int,
        nargs="*",
        default=None,
        help="Optional port filter list (e.g., --port-filter 8080).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving output artifacts to reports/.",
    )

    args = parser.parse_args()
    splits_to_eval = ["val", "test"] if args.split == "both" else [args.split]

    print("=" * 72)
    print("CyberSentry — Secondary Review Heuristic Evaluation")
    print("=" * 72)
    print(f"Splits: {splits_to_eval}")
    print(f"Port filter: {args.port_filter}")
    print()

    summaries = run_evaluation(
        splits=splits_to_eval,
        save_artifacts=not args.no_save,
        target_ports=args.port_filter,
    )

    for split_name, summary in summaries.items():
        print(f"--- Dataset Split: {split_name.upper()} ---")
        print(f"Total flows:                 {summary['total_flows']:,}")
        print(f"Total flagged for review:    {summary['total_flagged_flows']:,} ({summary['review_rate_pct']})")
        print(f"Botnet flows flagged:        {summary['botnet_flows_flagged']} / {summary['botnet_flows_total']} ({summary['botnet_flagging_rate_pct']})")
        print(f"Benign flows flagged (FPR):  {summary['benign_flows_flagged']} / {summary['benign_flows_total']} ({summary['benign_fpr_pct']})")
        print(f"Missing feature counts:      {summary['missing_or_unusable_features']}")
        print()

    if not args.no_save:
        print(f"Artifacts successfully written to: {REPORTS}")
        print(f"  - {REPORTS / 'secondary_review_metrics.json'}")
        print(f"  - {REPORTS / 'secondary_review_per_class.csv'}")


if __name__ == "__main__":
    main()
