"""Step 1 (v2): Class-aware split built from the existing processed data.

Because the raw CIC-IDS2017 CSV files are not available in this repository,
this script loads the already-cleaned processed splits (train/val/test parquet
or csv.gz files), concatenates them, applies a per-class stratified-block
split, and writes the new partitions to data/processed_v2/.

Key design decisions
--------------------
* No raw-CSV reader, no re-cleaning, no re-labelling.  The processed data
  already has clean float32 features, mapped labels, src_file, and row_in_file.
* Splitting unit is a consecutive block of ``block_size`` rows (sorted by
  src_file + row_in_file) per class.  Whole blocks are shuffled and assigned
  to train / val / test so adjacent flows from the same traffic session are
  kept together.
* The feature list and medians are recomputed from the new training partition
  and written to models/v2/meta.json.

WARNING — data provenance
--------------------------
The original processed test split was already examined during the v1
evaluation (XGBoost, RF, LR) and during the root-cause investigation of the
Botnet failure.  The new v2 split is derived by re-partitioning the same
underlying rows.  It is therefore NOT an independent, untouched test set.
Evaluation results from v2 models should be reported as:
  "Experiment B: representative evaluation on a re-partitioned dataset
   (same source rows as v1; original test already examined)."

Outputs
-------
data/processed_v2/{train,val,test}.parquet  (or .pkl if pyarrow is absent)
models/v2/meta.json
reports/v2/audit_v2.json
reports/v2/split_distribution_v2.csv
reports/v2/class_distribution_v2.png

Usage
-----
    python -m src.ml.prepare_data_v2
    python -m src.ml.prepare_data_v2 --block-size 500
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import META_COLS, PROC, SEED, load_table  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_ROOT = Path(__file__).resolve().parents[2]


def _v2_dirs(base: Path):
    return (
        base / "data" / "processed_v2",
        base / "models" / "v2",
        base / "reports" / "v2",
    )


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def stratified_block_split(
    df: pd.DataFrame,
    block_size: int,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = SEED,
) -> pd.Series:
    """Assign every row to train/val/test by shuffling consecutive blocks per class.

    For each class:
    1. Rows are sorted by (src_file, row_in_file) to respect capture order.
    2. They are divided into blocks of ``block_size`` consecutive rows.
    3. Whole blocks are randomly shuffled and assigned proportionally.

    This ensures all temporal sub-patterns within a class appear in every
    split, rather than only the earliest pattern appearing in training
    (which was the root cause of the Botnet failure in v1).

    Classes with fewer than 3 rows are placed entirely in training.
    Classes with 1 or 2 blocks (too few to make a 3-way split) have their
    rows split at the individual-row level to guarantee val/test presence.
    """
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index, dtype=object)

    for lab in df["label"].unique():
        mask = df["label"] == lab
        order = (
            df.loc[mask]
            .sort_values(["src_file", "row_in_file"])
            .index
        )
        n = len(order)

        if n < 3:
            # Cannot make a 3-way split — keep all in train
            continue

        n_blocks = int(np.ceil(n / block_size))

        if n_blocks < 3:
            # Row-level split to guarantee val/test presence
            a = max(1, int(n * ratios[0]))
            b = min(n - 1, int(n * (ratios[0] + ratios[1])))
            split.loc[order[:a]] = "train"
            split.loc[order[a:b]] = "val"
            split.loc[order[b:]] = "test"
            continue

        # Block-level shuffle
        block_order = np.arange(n_blocks)
        rng.shuffle(block_order)
        n_val = max(1, round(n_blocks * ratios[1]))
        n_test = max(1, round(n_blocks * ratios[2]))
        val_blocks = set(block_order[:n_val].tolist())
        test_blocks = set(block_order[n_val: n_val + n_test].tolist())

        for pos, row_idx in enumerate(order):
            blk = pos // block_size
            if blk in val_blocks:
                split.loc[row_idx] = "val"
            elif blk in test_blocks:
                split.loc[row_idx] = "test"
            # else stays "train"

    return split


# ---------------------------------------------------------------------------
# Data loading (from processed splits)
# ---------------------------------------------------------------------------

def load_processed_combined() -> pd.DataFrame:
    """Load all three processed splits and concatenate into one DataFrame.

    Uses the repository's established load_table() utility which handles
    parquet, pickle, and multi-part csv.gz formats.
    """
    print("Loading existing processed splits...")
    frames = []
    for sp in ("train", "val", "test"):
        part = load_table(sp)
        part["_original_split"] = sp   # for provenance audit; dropped later
        print(f"  {sp}: {len(part):,} rows")
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    print(f"  combined: {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_table_v2(df: pd.DataFrame, name: str, proc: Path) -> Path:
    proc.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        path = proc / f"{name}.parquet"
        df.to_parquet(path, index=False)
    except ImportError:
        path = proc / f"{name}.pkl"
        df.to_pickle(path)
    return path


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------

def check_exact_duplicate_leakage(
    splits: dict[str, pd.DataFrame],
    feat_cols: list[str],
) -> dict:
    """Count rows with identical feature vectors that appear in more than one split.

    Only exact feature-vector matches are checked.  Near-duplicate detection
    is not performed here because it would require an O(n^2) distance
    computation on ~2M rows.  A limitation note is recorded in the audit.
    """
    results: dict[str, int] = {}
    split_names = list(splits.keys())
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            da = splits[a][feat_cols].assign(_tag=a)
            db = splits[b][feat_cols].assign(_tag=b)
            merged = da.merge(db, on=feat_cols, how="inner")
            key = f"{a}_and_{b}_exact_feature_duplicates"
            results[key] = len(merged)
            if len(merged):
                print(f"  WARNING: {len(merged)} exact feature-vector duplicates "
                      f"between {a} and {b}")
            else:
                print(f"  OK: no exact feature-vector duplicates between {a} and {b}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block-size", type=int, default=1000,
                    help="Consecutive rows per block per class (default: 1000)")
    ap.add_argument("--also-csv", action="store_true",
                    help="Also write gzip-compressed CSV copies")
    args = ap.parse_args()

    base = Path(os.environ.get("CS_WORKDIR", CODE_ROOT))
    proc_v2, models_v2, reports_v2 = _v2_dirs(base)
    for p in (proc_v2, models_v2, reports_v2):
        p.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load from processed splits
    # ------------------------------------------------------------------
    df = load_processed_combined()

    # Identify feature columns (everything that's not label, meta, or provenance)
    non_feat = set(["label", "_original_split"] + META_COLS)
    feat = [c for c in df.columns if c not in non_feat]

    # Verify schema
    missing_meta = [c for c in META_COLS if c not in df.columns]
    if missing_meta:
        raise SystemExit(f"Processed data is missing required columns: {missing_meta}. "
                         "Re-run prepare_data.py to regenerate the processed files.")

    print(f"\nFeature columns: {len(feat)}")
    print(f"Labels: {sorted(df['label'].unique().tolist())}")
    print(f"Source files: {sorted(df['src_file'].unique().tolist())}")

    original_split_distribution = (
        df.groupby(["_original_split", "label"])
        .size()
        .unstack(fill_value=0)
        .to_dict()
    )

    # Drop provenance column before splitting
    df = df.drop(columns=["_original_split"])

    audit: dict = {
        "WARNING": (
            "The original test split was already examined during v1 evaluation "
            "and Botnet root-cause investigation. This v2 dataset is derived by "
            "re-partitioning those same rows. It is NOT an independent, untouched "
            "test set. Results must be reported as Experiment B: representative "
            "evaluation on a re-partitioned dataset."
        ),
        "experiment": "v2",
        "data_source": "existing processed splits (not raw CSVs)",
        "split_method": "stratified_block_per_class",
        "block_size": args.block_size,
        "seed": SEED,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "rows_combined": int(len(df)),
        "original_split_distribution": original_split_distribution,
        "source_files": sorted(df["src_file"].unique().tolist()),
    }

    # ------------------------------------------------------------------
    # 2. Apply stratified block split
    # ------------------------------------------------------------------
    print(f"\nApplying stratified block split (block_size={args.block_size}, seed={SEED})...")
    df["split"] = stratified_block_split(df, args.block_size)

    table = pd.crosstab(df["label"], df["split"]).reindex(
        columns=["train", "val", "test"], fill_value=0
    )
    table["total"] = table.sum(axis=1)
    table = table.sort_values("total", ascending=False)
    print("\nClass counts per split (v2):\n", table.to_string())

    for cls, row in table.iterrows():
        for sp in ("train", "val", "test"):
            if row[sp] == 0:
                print(f"WARNING: class '{cls}' has no rows in {sp}")

    audit["class_counts_per_split"] = {
        cls: {sp: int(row[sp]) for sp in ("train", "val", "test", "total")}
        for cls, row in table.iterrows()
    }

    # ------------------------------------------------------------------
    # 3. Feature audit for Botnet
    # ------------------------------------------------------------------
    print("\nBotnet feature distribution per split (v2):")
    botnet_audit = {}
    for sp in ("train", "val", "test"):
        part_b = df[(df["split"] == sp) & (df["label"] == "Botnet")]
        if len(part_b):
            info = {
                "count": int(len(part_b)),
                "zero_fwd_bytes_rate": round(float((part_b["Subflow Fwd Bytes"] == 0).mean()), 4),
                "median_subflow_fwd_bytes": float(part_b["Subflow Fwd Bytes"].median()),
                "median_max_packet_length": float(part_b["Max Packet Length"].median()),
                "row_in_file_min": int(part_b["row_in_file"].min()),
                "row_in_file_max": int(part_b["row_in_file"].max()),
            }
        else:
            info = {"count": 0}
        botnet_audit[sp] = info
        print(f"  {sp}: {info}")

    # Check if both subtypes (zero vs non-zero fwd bytes) appear in training
    train_b = df[(df["split"] == "train") & (df["label"] == "Botnet")]
    has_subtype_a = int((train_b["Subflow Fwd Bytes"] > 0).sum())
    has_subtype_b = int((train_b["Subflow Fwd Bytes"] == 0).sum())
    print(f"\n  Training Botnet: Subtype A (Fwd Bytes > 0): {has_subtype_a}, "
          f"Subtype B (Fwd Bytes = 0): {has_subtype_b}")
    if has_subtype_a == 0 or has_subtype_b == 0:
        print("  WARNING: one Botnet subtype is missing from training — "
              "try a smaller block size.")
    botnet_audit["training_subtype_a_count"] = has_subtype_a
    botnet_audit["training_subtype_b_count"] = has_subtype_b
    audit["botnet_feature_audit"] = botnet_audit

    # ------------------------------------------------------------------
    # 4. Constant-column check (on new training partition)
    # ------------------------------------------------------------------
    train_part = df[df["split"] == "train"]
    nun = train_part[feat].nunique()
    const = nun[nun <= 1].index.tolist()
    if const:
        print(f"\nDropping {len(const)} constant column(s) from new training partition: {const}")
    feat_final = [c for c in feat if c not in const]
    audit["constant_columns_dropped"] = const
    audit["n_features_final"] = len(feat_final)

    # ------------------------------------------------------------------
    # 5. Check exact-duplicate leakage across splits
    # ------------------------------------------------------------------
    print("\nChecking exact feature-vector duplicates across splits...")
    splits_dict = {
        sp: df[df["split"] == sp].reset_index(drop=True)
        for sp in ("train", "val", "test")
    }
    leakage = check_exact_duplicate_leakage(splits_dict, feat_final)
    audit["leakage_check"] = leakage
    audit["leakage_check_note"] = (
        "Only exact feature-vector matches are checked. Near-duplicate "
        "detection (e.g. L2-distance < epsilon) is not performed because "
        "it requires O(n^2) computation on ~2M rows. Small differences in "
        "float32 values between otherwise similar flows mean exact-match "
        "leakage is a lower bound on actual similarity."
    )

    # ------------------------------------------------------------------
    # 6. Save split partitions
    # ------------------------------------------------------------------
    keep = feat_final + ["label"] + META_COLS
    for sp in ("train", "val", "test"):
        part = df.loc[df["split"] == sp, keep].reset_index(drop=True)
        path = save_table_v2(part, sp, proc_v2)
        print(f"saved {path}")
        if args.also_csv:
            csv_path = proc_v2 / f"{sp}.csv.gz"
            part.to_csv(csv_path, index=False, float_format="%.9g", compression="gzip")
            print(f"saved {csv_path}")

    # ------------------------------------------------------------------
    # 7. Save meta.json for v2 models
    # ------------------------------------------------------------------
    medians = train_part[feat_final].median().to_dict()
    classes = ["Benign"] + sorted(c for c in df["label"].unique() if c != "Benign")
    meta = {
        "WARNING": audit["WARNING"],
        "experiment": "v2",
        "classes": classes,
        "benign_idx": 0,
        "features": feat_final,
        "medians": {k: float(v) for k, v in medians.items()},
        "block_size": args.block_size,
        "seed": SEED,
    }
    (models_v2 / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nMeta written to {models_v2 / 'meta.json'}")

    # ------------------------------------------------------------------
    # 8. Audit report
    # ------------------------------------------------------------------
    audit["class_counts_combined"] = df["label"].value_counts().to_dict()
    (reports_v2 / "audit_v2.json").write_text(json.dumps(audit, indent=2, default=str))
    table.to_csv(reports_v2 / "split_distribution_v2.csv")

    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values)
    ax.set_yscale("log")
    ax.set_ylabel("Flows (log scale)")
    ax.set_title("Class distribution — v2 stratified-block split")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(reports_v2 / "class_distribution_v2.png", dpi=150)
    plt.close(fig)

    print(f"\nAudit written to {reports_v2 / 'audit_v2.json'}")
    print(f"\n{'='*60}")
    print("IMPORTANT: See 'WARNING' key in audit_v2.json regarding data provenance.")
    print("v2 evaluation is REPRESENTATIVE, not independent temporal generalisation.")
    print("='*60}\n")
    print("Next: python -m src.ml.train_models --proc-dir data/processed_v2 "
          "--models-dir models/v2 --reports-dir reports/v2")


if __name__ == "__main__":
    main()
