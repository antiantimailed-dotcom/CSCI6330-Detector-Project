"""
Reliability and efficiency measurements for the Baseline (autoencoder) vs
RF-IDS comparison.

These are deliberately SEPARATE from each model's run_pilot():
  - Reliability trains several throwaway models per detector, one per
    random seed, and reports mean +/- std of val metrics. None of these
    runs are saved to models/, so they never touch the "official" saved
    model/threshold that run_pilot() produced, and they never clear or
    interact with the test-set lock files in autoencoder.py / rf_ids.py.
  - Efficiency times one train pass and one scoring pass for each
    detector, and reports the on-disk size of whatever model is
    currently saved in models/ (i.e. the official one from run_pilot()).

Only reads train/val — never touches test. Safe to re-run as many times
as you like.
"""

from pathlib import Path
import time
import numpy as np

from src.evaluate import (
    load_split, get_features_and_labels, pick_threshold_at_fpr,
    evaluate_at_threshold,
)

DEFAULT_SEEDS = [42, 7, 123]
METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "fpr", "fnr")


def _summarize(metric_dicts):
    summary = {}
    for k in METRIC_KEYS:
        vals = np.array([m[k] for m in metric_dicts], dtype=float)
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return summary


def _print_reliability(name, summary, seeds):
    print(f"\n=== {name} — reliability across seeds {seeds} ===")
    for k, v in summary.items():
        print(f"  {k:10s}: {v['mean']:.4f} +/- {v['std']:.4f}")


def _print_efficiency(name, result):
    print(f"\n=== {name} — efficiency ===")
    print(f"  train time       : {result['train_seconds']:.2f} s  ({result['n_train_rows']} rows)")
    print(f"  inference time   : {result['inference_seconds_total']:.4f} s total, "
          f"{result['inference_seconds_per_1000_rows']:.4f} s / 1000 rows  ({result['n_val_rows']} rows)")
    if result["saved_model_size_kb"] is not None:
        print(f"  saved model size : {result['saved_model_size_kb']:.1f} KB")
    else:
        print("  saved model size : (no saved artifact found — run run_pilot() first to produce one)")


# ------------------------------------------------------------------- data

def _load_xy(processed_dir):
    train_df = load_split(processed_dir, "train")
    val_df = load_split(processed_dir, "val")
    X_train, y_train = get_features_and_labels(train_df)
    X_val, y_val = get_features_and_labels(val_df)
    return X_train, y_train, X_val, y_val


# --------------------------------------------------------- Baseline (AE)

def _run_autoencoder_once(X_train_benign, X_val, y_val, target_fpr, seed):
    from src.autoencoder import fit_scaler, train_autoencoder, reconstruction_error
    scaler = fit_scaler(X_train_benign)
    X_train_scaled = scaler.transform(X_train_benign)
    X_val_scaled = scaler.transform(X_val)

    model = train_autoencoder(X_train_scaled, seed=seed)

    val_scores = reconstruction_error(model, X_val_scaled)
    benign_val_scores = val_scores[y_val == 0]
    threshold = pick_threshold_at_fpr(benign_val_scores, target_fpr=target_fpr)
    return evaluate_at_threshold(y_val, val_scores, threshold)


def reliability_autoencoder(processed_dir="data/processed", target_fpr=0.01, seeds=DEFAULT_SEEDS):
    """Train the autoencoder from scratch once per seed (never saved to
    models/), report mean +/- std of val metrics. Does NOT touch test."""
    X_train, y_train, X_val, y_val = _load_xy(processed_dir)
    X_train_benign = X_train[y_train == 0].to_numpy()
    X_val_np, y_val_np = X_val.to_numpy(), y_val.to_numpy()

    reports = []
    for seed in seeds:
        print(f"\n--- Baseline (autoencoder), seed={seed} ---")
        reports.append(_run_autoencoder_once(X_train_benign, X_val_np, y_val_np, target_fpr, seed))

    summary = _summarize(reports)
    _print_reliability("Baseline (autoencoder)", summary, seeds)
    return reports, summary


def measure_efficiency_autoencoder(processed_dir="data/processed", seed=42, models_dir="models"):
    """Times one train pass and one scoring pass; reports size of the
    model currently saved in models_dir (the one run_pilot() produced)."""
    from src.autoencoder import fit_scaler, train_autoencoder, reconstruction_error

    X_train, y_train, X_val, _ = _load_xy(processed_dir)
    X_train_benign = X_train[y_train == 0].to_numpy()
    X_val_np = X_val.to_numpy()

    scaler = fit_scaler(X_train_benign)
    X_train_scaled = scaler.transform(X_train_benign)
    X_val_scaled = scaler.transform(X_val_np)

    t0 = time.perf_counter()
    model = train_autoencoder(X_train_scaled, seed=seed)
    train_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = reconstruction_error(model, X_val_scaled)
    infer_seconds = time.perf_counter() - t0

    model_path = Path(models_dir) / "autoencoder.pt"
    size_kb = model_path.stat().st_size / 1024 if model_path.exists() else None

    result = {
        "train_seconds": train_seconds,
        "n_train_rows": len(X_train_benign),
        "inference_seconds_total": infer_seconds,
        "inference_seconds_per_1000_rows": infer_seconds / len(X_val_np) * 1000,
        "n_val_rows": len(X_val_np),
        "saved_model_size_kb": size_kb,
    }
    _print_efficiency("Baseline (autoencoder)", result)
    return result


# ------------------------------------------------------------------ RF-IDS

def _run_rf_once(X_train, y_train, X_val, y_val, target_fpr, seed):
    from src.rf_ids import train_rf, attack_probability
    rf = train_rf(X_train, y_train, seed=seed)
    val_scores = attack_probability(rf, X_val)
    benign_val_scores = val_scores[y_val == 0]
    threshold = pick_threshold_at_fpr(benign_val_scores, target_fpr=target_fpr)
    return evaluate_at_threshold(y_val, val_scores, threshold)


def reliability_rf(processed_dir="data/processed", target_fpr=0.01, seeds=DEFAULT_SEEDS):
    """Train RF-IDS from scratch once per seed (never saved to models/),
    report mean +/- std of val metrics. Does NOT touch test."""
    X_train, y_train, X_val, y_val = _load_xy(processed_dir)
    X_train_np, y_train_np = X_train.to_numpy(), y_train.to_numpy()
    X_val_np, y_val_np = X_val.to_numpy(), y_val.to_numpy()

    reports = []
    for seed in seeds:
        print(f"\n--- RF-IDS, seed={seed} ---")
        reports.append(_run_rf_once(X_train_np, y_train_np, X_val_np, y_val_np, target_fpr, seed))

    summary = _summarize(reports)
    _print_reliability("RF-IDS", summary, seeds)
    return reports, summary


def measure_efficiency_rf(processed_dir="data/processed", seed=42, n_estimators=200,
                           max_depth=None, models_dir="models"):
    """Times one train pass and one scoring pass; reports size of the
    model currently saved in models_dir (the one run_pilot() produced)."""
    from src.rf_ids import train_rf, attack_probability

    X_train, y_train, X_val, _ = _load_xy(processed_dir)
    X_train_np, y_train_np = X_train.to_numpy(), y_train.to_numpy()
    X_val_np = X_val.to_numpy()

    t0 = time.perf_counter()
    rf = train_rf(X_train_np, y_train_np, n_estimators=n_estimators, max_depth=max_depth, seed=seed)
    train_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = attack_probability(rf, X_val_np)
    infer_seconds = time.perf_counter() - t0

    model_path = Path(models_dir) / "rf_ids_model.joblib"
    size_kb = model_path.stat().st_size / 1024 if model_path.exists() else None

    result = {
        "train_seconds": train_seconds,
        "n_train_rows": len(X_train_np),
        "inference_seconds_total": infer_seconds,
        "inference_seconds_per_1000_rows": infer_seconds / len(X_val_np) * 1000,
        "n_val_rows": len(X_val_np),
        "saved_model_size_kb": size_kb,
    }
    _print_efficiency("RF-IDS", result)
    return result


# ------------------------------------------------------------------ runner

def run_all(processed_dir="data/processed", target_fpr=0.01, seeds=DEFAULT_SEEDS, models_dir="models"):
    """Runs reliability (multi-seed) and efficiency (timing + size) for
    both detectors. Call this once you have val-stage results you're
    happy with — it's a reporting step, not a tuning step, and it never
    touches test."""
    _, ae_summary = reliability_autoencoder(processed_dir, target_fpr, seeds)
    _, rf_summary = reliability_rf(processed_dir, target_fpr, seeds)
    ae_eff = measure_efficiency_autoencoder(processed_dir, seed=seeds[0], models_dir=models_dir)
    rf_eff = measure_efficiency_rf(processed_dir, seed=seeds[0], models_dir=models_dir)
    return {
        "autoencoder_reliability": ae_summary,
        "rf_reliability": rf_summary,
        "autoencoder_efficiency": ae_eff,
        "rf_efficiency": rf_eff,
    }


if __name__ == "__main__":
    run_all()
