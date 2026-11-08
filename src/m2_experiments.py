"""
Milestone 2 experiments: both detectors, trained on seen days (Mon-Wed),
evaluated on unseen attack families from Thursday and Friday.

Workflow
  1. run_val()           multi-seed validation report (train/val only).
                         Repeat as often as you like. Never touches a test file.
  2. save_official()     train the seed-42 models, freeze thresholds from
                         BENIGN VALIDATION scores, save to models_m2/ and
                         clear the test lock. (models/ from Milestone 1 is untouched.)
  3. score_test()        score test_seen and test_unseen ONCE with the frozen
                         models. Saves every per-flow score to
                         models_m2/test_scores_{seen,unseen}.csv.gz. A lock
                         file stops accidental re-scoring; force=True overrides.
  4. reanalyze_saved()   re-run all tables from the saved scores. Needs no
                         models, never re-scores, ignores the lock. Use this
                         (or load_test_scores) for any analysis after the test.

Operating points: strict `score > threshold` at the 99th and 99.9th benign
validation percentiles (target FPR 1% and 0.1%). Primary is 1%; 0.1% is a
sensitivity check. Because validation benign traffic comes from the same
hours as training, the FPR measured on the unseen days is the honest number.

Reported views
  RECALL / FPR at the FROZEN threshold ......... the headline results.
  AUROC overall, by day, by attack family ....... threshold-free ranking quality.
      Family AUROC compares a family with benign flows from the SAME day.
  RECALL with the threshold re-set on that day's own benign flows ...
      SECONDARY diagnostic: separates "cannot rank this family" from
      "the threshold drifted on that day". It uses test benign scores to
      set thresholds, so it is not a frozen-threshold result and must be
      labelled as such in the report.
Families with very few rows (Heartbleed, Infiltration, SQL Injection) are
descriptive only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluate import load_split, NON_FEATURE_COLS, pick_threshold_at_fpr

DATA_DIR = "data/processed_m2"
MODELS_DIR = Path("models_m2")
TARGET_FPRS = (0.01, 0.001)
DEFAULT_SEEDS = (42, 7, 123)
LOCK = "test_evaluated.json"


# ------------------------------------------------------------------ data

def _prep(df):
    """Features + labels, same filtering as evaluate.get_features_and_labels
    (drop rows with inf/NaN), but keeping the family label, the day tag and
    the original row index (idx) of every kept row."""
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1).to_numpy()
    return {
        "X": X[ok].to_numpy(dtype=float),
        "y": (df["binary_label"] == "ATTACK").astype(int).to_numpy()[ok],
        "label": df["Label"].astype(str).to_numpy()[ok],
        "day": df["day"].astype(str).to_numpy()[ok],
        "idx": np.flatnonzero(ok),
    }


def _load(name, data_dir=DATA_DIR):
    return _prep(load_split(data_dir, name))


def _slug(name):
    return name.lower().replace("-", "_")


# ---------------------------------------------------------------- models

def _make_dets(ae, scaler, rf):
    from src.autoencoder import reconstruction_error
    from src.rf_ids import attack_probability
    return {
        "Autoencoder": {"model": ae, "scaler": scaler,
                        "score": lambda X: reconstruction_error(ae, scaler.transform(X))},
        "RF-IDS": {"model": rf, "score": lambda X: attack_probability(rf, X)},
    }


def fit_detectors(train, seed):
    from src.autoencoder import fit_scaler, train_autoencoder
    from src.rf_ids import train_rf
    Xb = train["X"][train["y"] == 0]
    scaler = fit_scaler(Xb)
    ae = train_autoencoder(scaler.transform(Xb), seed=seed)
    rf = train_rf(train["X"], train["y"], seed=seed)
    return _make_dets(ae, scaler, rf)


def freeze_thresholds(dets, val):
    """Thresholds from BENIGN validation scores only, per target FPR."""
    thr = {}
    for name, d in dets.items():
        s = d["score"](val["X"])
        thr[name] = {f: pick_threshold_at_fpr(s[val["y"] == 0], f) for f in TARGET_FPRS}
    return thr


# ------------------------------------------------------------ evaluation

def score_all(dets, data):
    """Per-flow scores for every detector: {name: array}."""
    return {name: d["score"](data["X"]) for name, d in dets.items()}


def evaluate_split(dets, thr, data, scores=None):
    """Long-format results (strict > flagging). Pass `scores` to reuse
    precomputed per-flow scores; then `dets` and data['X'] are not needed."""
    y, label, day = data["y"], data["label"], data["day"]
    rows = []

    def add(name, f, scope, item, n, value):
        rows.append(dict(detector=name, target_fpr=f, scope=scope, item=item, n=int(n), value=float(value)))

    for name in thr:
        s = scores[name] if scores is not None else dets[name]["score"](data["X"])

        # --- threshold-free views
        if 0 < y.sum() < len(y):
            add(name, np.nan, "auroc", "overall", len(y), roc_auc_score(y, s))
        for dk in sorted(set(day)):
            m = day == dk
            if 0 < y[m].sum() < m.sum():
                add(name, np.nan, "auroc_day", dk, m.sum(), roc_auc_score(y[m], s[m]))

        # --- per family: AUROC vs same-day benign, and day-rethresholded recall
        for lab in sorted(set(label[y == 1])):
            fam = label == lab
            same_day_benign = (y == 0) & np.isin(day, np.unique(day[fam]))
            if same_day_benign.sum() == 0:
                continue
            sel = fam | same_day_benign
            add(name, np.nan, "auroc_family", lab, fam.sum(), roc_auc_score(fam[sel].astype(int), s[sel]))
            for f in TARGET_FPRS:
                t_day = pick_threshold_at_fpr(s[same_day_benign], f)
                add(name, f, "recall_rethresh", lab, fam.sum(), (s[fam] > t_day).mean())

        # --- headline: frozen thresholds
        for f in TARGET_FPRS:
            flag = s > thr[name][f]
            if (y == 1).any():
                add(name, f, "recall", "ALL attacks", (y == 1).sum(), flag[y == 1].mean())
            if (y == 0).any():
                add(name, f, "fpr", "ALL benign", (y == 0).sum(), flag[y == 0].mean())
            for lab in sorted(set(label[y == 1])):
                m = label == lab
                add(name, f, "recall", lab, m.sum(), flag[m].mean())
            for dk in sorted(set(day[y == 0])):
                m = (day == dk) & (y == 0)
                add(name, f, "fpr", dk, m.sum(), flag[m].mean())
    return pd.DataFrame(rows)


_SPECS = [
    ("auroc", "AUROC, all flows", "plain"),
    ("auroc_day", "AUROC by day (that day's attacks vs that day's benign)", "plain"),
    ("auroc_family", "AUROC by attack family (family vs benign flows from the same day)", "plain"),
    ("recall", "RECALL by attack family at the FROZEN threshold (headline)", "tgt"),
    ("fpr", "FPR (benign flows flagged) by day at the FROZEN threshold", "tgt"),
    ("recall_rethresh", "RECALL by family, threshold RE-SET on that day's own benign flows "
                        "(secondary diagnostic, NOT a frozen-threshold result)", "tgt"),
]


def show(df, title):
    print(f"\n=== {title} ===")
    for scope, head, kind in _SPECS:
        sub = df[df.scope == scope]
        if sub.empty:
            continue
        n = sub.groupby("item")["n"].first()
        if kind == "plain":
            tab = sub.pivot_table(index="item", columns="detector", values="value")
        else:
            tab = sub.pivot_table(index="item", columns=["detector", "target_fpr"], values="value")
            tab.columns = [f"{d} @{f:.1%}" for d, f in tab.columns]
        tab.insert(0, "n", n)
        print(f"\n{head}")
        print(tab.to_string(float_format=lambda v: f"{v:.4f}"))


# ------------------------------------------------------------ validation

def run_val(seeds=DEFAULT_SEEDS, data_dir=DATA_DIR):
    """Train both detectors per seed on train, freeze thresholds on val
    benign, report val metrics per seed and mean +/- std across seeds.
    Models are throwaway (not saved). Never reads a test file."""
    train, val = _load("train", data_dir), _load("val", data_dir)
    per_seed = []
    for seed in seeds:
        print(f"\n########## seed {seed} ##########")
        dets = fit_detectors(train, seed)
        thr = freeze_thresholds(dets, val)
        res = evaluate_split(dets, thr, val)
        res["seed"] = seed
        per_seed.append(res)
        show(res, f"validation, seed {seed}")
    allres = pd.concat(per_seed, ignore_index=True)
    rec = allres[allres.scope == "recall"]
    summ = (rec.groupby(["detector", "target_fpr", "item"])["value"].agg(["mean", "std"]).reset_index())
    print("\n=== validation recall across seeds (mean, std) ===")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nReminder: partitions are fixed across seeds, so std excludes split variability,")
    print("and same-seed AE recall also varies run to run (PyTorch non-determinism).")
    return allres, summ


# ----------------------------------------------------------- official run

def save_official(seed=42, data_dir=DATA_DIR, models_dir=MODELS_DIR):
    """Train seed-42 models, freeze thresholds on val benign, save to
    models_m2/, clear the test lock. Never reads a test file."""
    import torch, joblib
    models_dir = Path(models_dir); models_dir.mkdir(exist_ok=True)
    train, val = _load("train", data_dir), _load("val", data_dir)
    dets = fit_detectors(train, seed)
    thr = freeze_thresholds(dets, val)
    torch.save(dets["Autoencoder"]["model"], models_dir / "autoencoder.pt")
    joblib.dump(dets["Autoencoder"]["scaler"], models_dir / "scaler.joblib")
    joblib.dump(dets["RF-IDS"]["model"], models_dir / "rf.joblib")
    (models_dir / "thresholds.json").write_text(json.dumps(
        {n: {str(f): v for f, v in t.items()} for n, t in thr.items()}, indent=2))
    (models_dir / LOCK).unlink(missing_ok=True)
    show(evaluate_split(dets, thr, val), f"validation, official seed {seed}")
    print(f"\nSaved models and thresholds to {models_dir}/ ; test lock cleared.")
    return dets, thr


def _load_thr(models_dir):
    raw = json.loads((Path(models_dir) / "thresholds.json").read_text())
    return {n: {float(f): v for f, v in t.items()} for n, t in raw.items()}


def load_official(models_dir=MODELS_DIR):
    import torch, joblib
    models_dir = Path(models_dir)
    ae = torch.load(models_dir / "autoencoder.pt", map_location="cpu", weights_only=False)
    ae.eval()
    dets = _make_dets(ae, joblib.load(models_dir / "scaler.joblib"), joblib.load(models_dir / "rf.joblib"))
    return dets, _load_thr(models_dir)


# ---------------------------------------------------------------- scoring

def _save_scores(data, scores, path):
    df = pd.DataFrame({"idx": data["idx"], "y": data["y"], "label": data["label"], "day": data["day"]})
    for name, s in scores.items():
        df[f"score_{_slug(name)}"] = s
    df.to_csv(path, index=False, compression="gzip")


def _score_test_core(dets, thr, data_dir=DATA_DIR, scores_dir=None):
    out = {}
    for key, split in (("seen", "test_seen"), ("unseen", "test_unseen")):
        data = _load(split, data_dir)
        scores = score_all(dets, data)
        if scores_dir is not None:
            _save_scores(data, scores, Path(scores_dir) / f"test_scores_{key}.csv.gz")
        out[key] = evaluate_split(None, thr, data, scores=scores)
    show(out["seen"], "TEST, seen days (Mon-Wed held-out slice)")
    show(out["unseen"], "TEST, unseen days and attack families (Thu/Fri)")
    return out["seen"], out["unseen"]


def score_test(data_dir=DATA_DIR, models_dir=MODELS_DIR, force=False):
    """Score test_seen and test_unseen ONCE with the frozen official models,
    save all per-flow scores, write the results CSVs and the lock.
    Refuses to run twice unless force=True (save_official clears the lock)."""
    models_dir = Path(models_dir)
    lock = models_dir / LOCK
    if lock.exists() and not force:
        raise RuntimeError(f"Test already scored ({lock}). Re-scoring after seeing test results "
                           f"defeats the held-out protocol. Use reanalyze_saved() for further analysis, "
                           f"or force=True only if you must, and say so.")
    dets, thr = load_official(models_dir)
    res_seen, res_unseen = _score_test_core(dets, thr, data_dir, scores_dir=models_dir)
    res_seen.assign(split="test_seen").to_csv(models_dir / "test_results_seen.csv", index=False)
    res_unseen.assign(split="test_unseen").to_csv(models_dir / "test_results_unseen.csv", index=False)
    lock.write_text(json.dumps({"scored": True}))
    print(f"\nPer-flow scores saved to {models_dir}/test_scores_*.csv.gz")
    print(f"Results saved to {models_dir}/test_results_*.csv ; lock written.")
    return res_seen, res_unseen


# ------------------------------------------------- analysis after the test

def load_test_scores(models_dir=MODELS_DIR):
    """Saved per-flow test scores: {'seen': df, 'unseen': df} with columns
    idx (row in the original test CSV), y, label, day, score_<detector>."""
    d = Path(models_dir)
    return {k: pd.read_csv(d / f"test_scores_{k}.csv.gz") for k in ("seen", "unseen")}


def reanalyze_saved(models_dir=MODELS_DIR):
    """Rebuild every test table from the saved scores and frozen thresholds.
    Needs no models, never scores anything, does not touch the lock."""
    thr = _load_thr(models_dir)
    out = {}
    for key, df in load_test_scores(models_dir).items():
        data = {"y": df["y"].to_numpy(), "label": df["label"].astype(str).to_numpy(),
                "day": df["day"].astype(str).to_numpy()}
        scores = {n: df[f"score_{_slug(n)}"].to_numpy() for n in thr}
        out[key] = evaluate_split(None, thr, data, scores=scores)
        show(out[key], f"TEST, {key} (from saved scores)")
    return out
