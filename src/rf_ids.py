"""
RF-IDS: supervised Random Forest detector.

Trains on ALL of `train` (benign + attack) — unlike the Baseline autoencoder,
which only ever sees benign rows. To keep the comparison fair, RF-IDS is NOT
evaluated at its default 0.5 probability cutoff. Instead it's scored the
same way as the Baseline: predict_proba()'s attack probability is treated as
a continuous score, a threshold is picked from BENIGN validation scores at
the same target FPR, and that frozen threshold is what's actually used —
see "matched FPR" in src/evaluate.py's docstring for why this matters.

Requires: pip install scikit-learn joblib
"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.evaluate import (
    load_split, get_features_and_labels, pick_threshold_at_fpr,
    evaluate_at_threshold, print_report, get_labels_aligned, per_attack_recall,
)

MODELS_DIR = Path("models")  # gitignored — regenerate locally, like the autoencoder's artifacts


def train_rf(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 200,
             max_depth=None, seed: int = 42) -> RandomForestClassifier:
    """Train on ALL of train (benign + attack) — this is RF-IDS's whole point,
    contrasted with the Baseline's benign-only training."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
        class_weight=None,  # revisit if class imbalance tuning is wanted later
    )
    rf.fit(X_train, y_train)
    return rf


def attack_probability(rf: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """Continuous attack-probability score — the RF analogue of the
    autoencoder's reconstruction error. Higher = more likely an attack."""
    return rf.predict_proba(X)[:, 1]


def save_artifacts(rf: RandomForestClassifier, threshold: float, models_dir: Path = MODELS_DIR):
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, models_dir / "rf_ids_model.joblib")
    with open(models_dir / "rf_ids_meta.json", "w") as f:
        json.dump({"threshold": threshold}, f)

    # Same reasoning as the autoencoder: a freshly trained model hasn't been
    # test-scored yet, so clear any stale lock from a previous model version.
    lock_path = models_dir / "rf_ids_test_evaluated.json"
    if lock_path.exists():
        lock_path.unlink()
        print("Cleared previous test-evaluation lock (new model saved, not yet test-scored).")

    print(f"Saved RF-IDS model and threshold to {models_dir}/")


def load_artifacts(models_dir: Path = MODELS_DIR):
    rf = joblib.load(models_dir / "rf_ids_model.joblib")
    with open(models_dir / "rf_ids_meta.json") as f:
        meta = json.load(f)
    return rf, meta["threshold"]


def run_pilot(processed_dir: str = "data/processed", target_fpr: float = 0.01,
              models_dir: Path = MODELS_DIR, n_estimators: int = 200, max_depth=None):
    """
    Milestone-1 pilot run: train on all of train (benign + attack), pick a
    MATCHED-FPR threshold on val-benign scores (not RF's default cutoff),
    report metrics on val (NOT test). Saves the trained model/threshold.
    """
    train_df = load_split(processed_dir, "train")
    val_df = load_split(processed_dir, "val")

    X_train, y_train = get_features_and_labels(train_df)
    X_val, y_val = get_features_and_labels(val_df)

    print(f"Training on {len(X_train)} rows ({y_train.sum()} attack, "
          f"{len(X_train) - y_train.sum()} benign)")

    rf = train_rf(X_train.to_numpy(), y_train.to_numpy(),
                  n_estimators=n_estimators, max_depth=max_depth)

    val_scores = attack_probability(rf, X_val.to_numpy())
    benign_val_scores = val_scores[y_val.to_numpy() == 0]
    threshold = pick_threshold_at_fpr(benign_val_scores, target_fpr=target_fpr)
    print(f"\nThreshold set at {target_fpr:.1%} target FPR on benign val scores "
          f"(matched to Baseline's protocol, not RF's default 0.5 cutoff): {threshold:.6f}")

    metrics = evaluate_at_threshold(y_val.to_numpy(), val_scores, threshold)
    print_report("RF-IDS — val split (pilot checkpoint)", metrics)
    print("\nPer-attack recall (val):")
    print(per_attack_recall(get_labels_aligned(val_df), val_scores, threshold).to_string())

    save_artifacts(rf, threshold, models_dir)
    return rf, threshold, metrics


def score_test(processed_dir: str = "data/processed", models_dir: Path = MODELS_DIR,
                force: bool = False):
    """Run the FROZEN model/threshold on the test split. Call this only once,
    for the final comparison — not during iteration/tuning. Same one-time
    guard as src/autoencoder.py's score_test(): refuses a second run against
    the same saved model unless force=True."""
    lock_path = Path(models_dir) / "rf_ids_test_evaluated.json"
    if lock_path.exists() and not force:
        prior = json.loads(lock_path.read_text())
        raise RuntimeError(
            f"Test set was already scored for this saved RF-IDS model at {prior['timestamp']} "
            f"(metrics: {prior['metrics']}).\n"
            "Scoring test again now would mean using test feedback to influence a model "
            "you're still iterating on — exactly the leakage the instructor flagged.\n"
            "If you have genuinely retrained/replaced the model since then and this is "
            "an intentional final re-evaluation, call score_test(force=True)."
        )

    rf, threshold = load_artifacts(models_dir)
    test_df = load_split(processed_dir, "test")
    X_test, y_test = get_features_and_labels(test_df)

    test_scores = attack_probability(rf, X_test.to_numpy())
    metrics = evaluate_at_threshold(y_test.to_numpy(), test_scores, threshold)
    print_report("RF-IDS — TEST (final)", metrics)
    print("\nPer-attack recall (TEST, final):")
    print(per_attack_recall(get_labels_aligned(test_df), test_scores, threshold).to_string())

    lock_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in metrics.items()},
    }, indent=2))
    print(f"\nRecorded this test run at {lock_path} — re-running score_test() will now "
          f"refuse unless you pass force=True.")

    return metrics


if __name__ == "__main__":
    run_pilot()
