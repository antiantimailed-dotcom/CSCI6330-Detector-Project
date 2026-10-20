"""
Shared evaluation for the Baseline (autoencoder) vs RF-IDS comparison.
Both detectors must be scored through this module so every number you
report is computed identically for each — same feature set, same metrics,
same matched-FPR protocol. Do not write a second, model-specific version
of any of this for either detector.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

# Must match DROP_FROM_FEATURES in data_prep.py plus the label/bookkeeping
# columns that aren't model inputs.
#
# "Fwd Header Length.1" is not a real feature — CICIDS2017's source CSV
# contains the column "Fwd Header Length" twice under the same name, and
# pandas auto-renames the second occurrence to "Fwd Header Length.1" on
# read. Both hold identical values, so we drop the duplicate and keep only
# the first "Fwd Header Length".
NON_FEATURE_COLS = ["Label", "binary_label", "day", "Destination Port", "Fwd Header Length.1"]


def load_split(processed_dir: str = "data/processed", name: str = "train") -> pd.DataFrame:
    """Load one of the train/val/test CSVs written by data_prep.run_pilot_split."""
    return pd.read_csv(Path(processed_dir) / f"{name}.csv", low_memory=False)


def get_features_and_labels(df: pd.DataFrame):
    """
    Split a loaded dataframe into:
      X — numeric feature matrix, with label/bookkeeping columns removed
      y — binary labels, 1 = ATTACK, 0 = BENIGN

    Replaces inf values (a few CICIDS2017 rate features can produce these
    from divide-by-zero) with NaN, then drops any row containing a NaN in
    either X or y, keeping X and y aligned.
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = (df["binary_label"] == "ATTACK").astype(int)

    valid = X.notna().all(axis=1)
    dropped = (~valid).sum()
    if dropped:
        print(f"  dropped {dropped} rows with inf/NaN feature values")

    return X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)


def pick_threshold_at_fpr(benign_scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """
    Pick a score threshold from BENIGN validation scores only, so that
    roughly target_fpr fraction of benign flows would be (wrongly) flagged.
    Scores must be oriented so higher = more likely an attack (e.g.
    reconstruction error for the autoencoder, predicted attack probability
    for RF-IDS). Freeze the returned threshold and reuse it as-is on test.
    """
    percentile = 100 * (1 - target_fpr)
    return float(np.percentile(benign_scores, percentile))


def evaluate_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Core shared metric: given any model's scores and a frozen threshold,
    compute the same metric set every detector in this project is judged by."""
    y_pred = (scores >= threshold).astype(int)
    return _metrics_from_predictions(y_true, y_pred, threshold=threshold)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Same metric set, for a model that only gives hard 0/1 predictions
    (e.g. RF-IDS at its default decision rule) rather than a continuous score."""
    return _metrics_from_predictions(y_true, y_pred, threshold=None)


def _metrics_from_predictions(y_true, y_pred, threshold) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fpr,
        "fnr": fnr,
        "n": len(y_true),
        "n_attack": int(np.sum(y_true)),
        "n_benign": int(len(y_true) - np.sum(y_true)),
    }


def print_report(name: str, metrics: dict):
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:10s}: {v:.4f}")
        else:
            print(f"  {k:10s}: {v}")


def compare_reports(reports: dict):
    """reports: {model_name: metrics_dict}. Prints a side-by-side table
    for the metrics that matter for the comparison (reliability/efficiency
    extras like runtime go in your own timing code, not here)."""
    cols = ["accuracy", "precision", "recall", "f1", "fpr", "fnr"]
    names = list(reports.keys())
    header = f"{'metric':10s}" + "".join(f"{n:>14s}" for n in names)
    print(header)
    print("-" * len(header))
    for c in cols:
        row = f"{c:10s}" + "".join(f"{reports[n][c]:14.4f}" for n in names)
        print(row)
