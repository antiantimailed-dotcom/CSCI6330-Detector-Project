"""
Baseline detector: a benign-only autoencoder.

Trains only on BENIGN rows from the train split. Reconstruction error is the
anomaly score — higher error means the row looks less like the benign
traffic the model learned from. The detection threshold is picked from
BENIGN validation scores only (never from attack rows, never from test),
then frozen and reused as-is. Scoring/evaluation goes through
src/evaluate.py so Baseline and RF-IDS are judged identically.

Requires: pip install torch scikit-learn
"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

from src.evaluate import (
    load_split, get_features_and_labels, pick_threshold_at_fpr,
    evaluate_at_threshold, print_report, get_labels_aligned, per_attack_recall,
)

MODELS_DIR = Path("models")  # gitignored — regenerate locally, like data/processed


class Autoencoder(nn.Module):
    """Dense autoencoder for tabular flow features.
    input_dim -> 64 -> 32 -> bottleneck -> 32 -> 64 -> input_dim
    """
    def __init__(self, input_dim: int, bottleneck: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),  # linear output — reconstructing scaled real values
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_train_benign_scaled: np.ndarray, bottleneck: int = 16,
                       epochs: int = 30, batch_size: int = 256, lr: float = 1e-3,
                       seed: int = 42) -> Autoencoder:
    """Train on benign-only rows. X_train_benign_scaled must already be scaled
    (see fit_scaler) and contain ONLY benign rows from the train split."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(input_dim=X_train_benign_scaled.shape[1], bottleneck=bottleneck).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X_train_benign_scaled, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  train MSE: {avg_loss:.6f}")

    model.eval()
    return model


def reconstruction_error(model: Autoencoder, X_scaled: np.ndarray) -> np.ndarray:
    """Per-row reconstruction error (mean squared error across features).
    This is the anomaly score — higher = more likely an attack."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        recon = model(X_tensor)
        errors = ((X_tensor - recon) ** 2).mean(dim=1)
    return errors.cpu().numpy()


def fit_scaler(X_train_benign: np.ndarray) -> StandardScaler:
    """Fit on benign train rows only — scaling parameters must not see attacks
    or val/test data, same principle as the model itself."""
    scaler = StandardScaler()
    scaler.fit(X_train_benign)
    return scaler


def save_artifacts(model: Autoencoder, scaler: StandardScaler, threshold: float,
                    models_dir: Path = MODELS_DIR):
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / "autoencoder.pt")
    joblib.dump(scaler, models_dir / "autoencoder_scaler.joblib")
    with open(models_dir / "autoencoder_meta.json", "w") as f:
        json.dump({"threshold": threshold, "input_dim": scaler.n_features_in_}, f)

    # A freshly trained/saved model hasn't been test-scored yet — clear any
    # stale lock from a previous model version so score_test()'s one-time
    # guard applies to THIS model, not whatever was there before.
    lock_path = models_dir / "autoencoder_test_evaluated.json"
    if lock_path.exists():
        lock_path.unlink()
        print("Cleared previous test-evaluation lock (new model saved, not yet test-scored).")

    print(f"Saved model, scaler, threshold to {models_dir}/")


def load_artifacts(models_dir: Path = MODELS_DIR):
    with open(models_dir / "autoencoder_meta.json") as f:
        meta = json.load(f)
    model = Autoencoder(input_dim=meta["input_dim"])
    model.load_state_dict(torch.load(models_dir / "autoencoder.pt"))
    model.eval()
    scaler = joblib.load(models_dir / "autoencoder_scaler.joblib")
    return model, scaler, meta["threshold"]


def run_pilot(processed_dir: str = "data/processed", target_fpr: float = 0.01,
              models_dir: Path = MODELS_DIR):
    """
    Milestone-1 pilot run: train on train-benign, pick threshold on val-benign,
    report metrics on val (NOT test — test stays untouched until the final
    comparison). Saves the trained model/scaler/threshold for reuse.
    """
    train_df = load_split(processed_dir, "train")
    val_df = load_split(processed_dir, "val")

    X_train, y_train = get_features_and_labels(train_df)
    X_val, y_val = get_features_and_labels(val_df)

    X_train_benign = X_train[y_train == 0].to_numpy()
    print(f"Training on {len(X_train_benign)} benign rows "
          f"(train split has {len(X_train)} rows total, {y_train.sum()} attack)")

    scaler = fit_scaler(X_train_benign)
    X_train_benign_scaled = scaler.transform(X_train_benign)
    X_val_scaled = scaler.transform(X_val.to_numpy())

    print("Training autoencoder...")
    model = train_autoencoder(X_train_benign_scaled)

    val_scores = reconstruction_error(model, X_val_scaled)
    benign_val_scores = val_scores[y_val.to_numpy() == 0]
    threshold = pick_threshold_at_fpr(benign_val_scores, target_fpr=target_fpr)
    print(f"\nThreshold set at {target_fpr:.1%} target FPR on benign val scores: {threshold:.6f}")

    metrics = evaluate_at_threshold(y_val.to_numpy(), val_scores, threshold)
    print_report("Baseline (autoencoder) — val split (pilot checkpoint)", metrics)
    print("\nPer-attack recall (val):")
    print(per_attack_recall(get_labels_aligned(val_df), val_scores, threshold).to_string())

    save_artifacts(model, scaler, threshold, models_dir)
    return model, scaler, threshold, metrics


def score_test(processed_dir: str = "data/processed", models_dir: Path = MODELS_DIR,
                force: bool = False):
    """Run the FROZEN model/threshold on the test split. Call this only once,
    for the final comparison — not during iteration/tuning.

    Refuses to run a second time against the same saved model (tracked by a
    lock file next to the model artifacts) unless force=True, so a restart,
    a re-run of "all cells", or forgetting it already ran can't quietly turn
    test into a second validation set. Pass force=True only when you
    deliberately intend a fresh, final re-evaluation (e.g. after genuinely
    retraining the model) — not to brush past the warning."""
    lock_path = Path(models_dir) / "autoencoder_test_evaluated.json"
    if lock_path.exists() and not force:
        prior = json.loads(lock_path.read_text())
        raise RuntimeError(
            f"Test set was already scored for this saved model at {prior['timestamp']} "
            f"(metrics: {prior['metrics']}).\n"
            "Scoring test again now would mean using test feedback to influence a model "
            "you're still iterating on — exactly the leakage the instructor flagged.\n"
            "If you have genuinely retrained/replaced the model since then and this is "
            "an intentional final re-evaluation, call score_test(force=True)."
        )

    model, scaler, threshold = load_artifacts(models_dir)
    test_df = load_split(processed_dir, "test")
    X_test, y_test = get_features_and_labels(test_df)
    X_test_scaled = scaler.transform(X_test.to_numpy())

    test_scores = reconstruction_error(model, X_test_scaled)
    metrics = evaluate_at_threshold(y_test.to_numpy(), test_scores, threshold)
    print_report("Baseline (autoencoder) — TEST (final)", metrics)
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
