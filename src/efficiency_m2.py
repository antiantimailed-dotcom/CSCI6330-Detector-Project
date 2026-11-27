import gc
import json
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from src import m2_experiments as m2

RESULTS_PATH = Path("results_m2") / "efficiency_m2.json"   # committed (models_m2/ is gitignored)


# --------------------------------------------------------------- memory

class PeakMemory:
    """Context manager: peak resident memory (MB) above the level at entry."""

    def __init__(self, interval=0.01):
        self.interval = interval
        self.proc = psutil.Process()

    def _watch(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        gc.collect()
        self.start = self.proc.memory_info().rss
        self.peak = self.start
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)
        self.peak_mb = (self.peak - self.start) / 1e6
        return False


def _timed(fn):
    """Run fn once; return (result, seconds, peak extra MB)."""
    with PeakMemory() as mem:
        t0 = time.perf_counter()
        out = fn()
        secs = time.perf_counter() - t0
    return out, secs, mem.peak_mb


def _stats(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std()), "runs": [float(x) for x in v]}


# ------------------------------------------------------------- training

def time_training(train, seed=42, repeats=3):
    from src.autoencoder import fit_scaler, train_autoencoder
    from src.rf_ids import train_rf

    X_benign = train["X"][train["y"] == 0]

    def fit_ae():
        scaler = fit_scaler(X_benign)
        return train_autoencoder(scaler.transform(X_benign), seed=seed)

    def fit_rf():
        return train_rf(train["X"], train["y"], seed=seed)

    out = {}
    for name, fn in (("Autoencoder", fit_ae), ("RF-IDS", fit_rf)):
        secs, mems = [], []
        for i in range(repeats):
            model, s, mb = _timed(fn)
            del model
            gc.collect()
            secs.append(s); mems.append(mb)
            print(f"  train {name:11s} run {i + 1}/{repeats}: {s:7.1f} s, peak +{mb:,.0f} MB")
        out[name] = {"seconds": _stats(secs), "peak_mb": _stats(mems),
                     "rows": int(len(X_benign) if name == "Autoencoder" else len(train["X"]))}
    return out


# ------------------------------------------------------------ inference

def time_inference(val, repeats=5):
    dets, _ = m2.load_official()
    n = len(val["X"])
    out = {}
    for name, d in dets.items():
        d["score"](val["X"])  # warm-up, discarded
        secs, mems = [], []
        for i in range(repeats):
            _, s, mb = _timed(lambda: d["score"](val["X"]))
            secs.append(s); mems.append(mb)
        per_1000 = [s / n * 1000 for s in secs]
        out[name] = {"seconds_total": _stats(secs), "seconds_per_1000_rows": _stats(per_1000),
                     "peak_mb": _stats(mems), "rows": n}
        print(f"  infer {name:11s}: {np.mean(per_1000):.5f} +/- {np.std(per_1000):.5f} s per 1,000 rows, "
              f"peak +{np.mean(mems):,.0f} MB")
    return out


# ----------------------------------------------------------- model size

def model_sizes(models_dir=m2.MODELS_DIR):
    d = Path(models_dir)
    kb = lambda f: (d / f).stat().st_size / 1024
    return {"Autoencoder": {"kb": kb("autoencoder.pt") + kb("scaler.joblib")},
            "RF-IDS": {"kb": kb("rf.joblib")}}


# ------------------------------------------------------------------ run

def run(train_repeats=3, infer_repeats=5, save=True):
    print("Loading M2 train/val (never test)...")
    train, val = m2._load("train"), m2._load("val")

    print(f"\nTraining time and memory ({train_repeats} runs each, throwaway models, nothing saved):")
    training = time_training(train, repeats=train_repeats)

    print(f"\nInference time and memory (frozen models, 1 warm-up + {infer_repeats} runs):")
    inference = time_inference(val, repeats=infer_repeats)

    sizes = model_sizes()
    print("\n=== Summary ===")
    for name in ("Autoencoder", "RF-IDS"):
        tr, inf = training[name], inference[name]
        print(f"{name}:")
        print(f"  training      : {tr['seconds']['mean']:.1f} +/- {tr['seconds']['std']:.1f} s "
              f"(peak +{tr['peak_mb']['mean']:,.0f} MB)")
        print(f"  inference     : {inf['seconds_per_1000_rows']['mean']:.5f} +/- "
              f"{inf['seconds_per_1000_rows']['std']:.5f} s per 1,000 rows "
              f"(peak +{inf['peak_mb']['mean']:,.0f} MB)")
        print(f"  model size    : {sizes[name]['kb']:,.0f} KB")

    result = {"training": training, "inference": inference, "model_size": sizes,
              "train_repeats": train_repeats, "infer_repeats": infer_repeats,
              "cpu_count": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 1e9, 1)}
    if save:
        RESULTS_PATH.parent.mkdir(exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(result, indent=2))
        print(f"\nSaved {RESULTS_PATH}")
    return result


if __name__ == "__main__":
    run()