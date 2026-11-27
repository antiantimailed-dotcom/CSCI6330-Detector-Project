"""
Pre-test diagnostics for the Baseline (autoencoder) vs RF-IDS comparison.

Written in response to the Milestone 1 feedback. Covers:
  1. list_inputs       - exact Monday/Wednesday files, retained features, split sizes
  2. port_only_check   - Destination Port alone: AUROC + recall at 1/5/10/20% FPR,
                         plus a tie diagnostic (is "0% recall at 1% FPR" a tie artifact?)
  3. context_ablation  - both models with vs. without Destination Port
  4. rf_leakage_check  - RF feature importances, retrain without top-k features,
                         exact-duplicate and nearest-neighbour distance checks
  5. error_analysis    - which flows the AE misses / the RF falsely flags

Reads train and val ONLY. Never touches test.csv and never calls score_test(),
so it cannot trip or clear the test-set lock files. Models trained here are
throwaway: nothing is written to models/.

All flagging uses the strict rule `score > threshold` (same as the official
pipeline). This file does not depend on evaluate_at_threshold, so it behaves
the same whether or not your evaluate.py has the `>` fix.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.evaluate import load_split, NON_FEATURE_COLS, pick_threshold_at_fpr

FPR_GRID = (0.01, 0.05, 0.10, 0.20)


# ------------------------------------------------------------------ helpers

def _load_aligned(df):
    """Same row filtering as evaluate.get_features_and_labels (drop rows with
    inf/NaN in any feature), but ALSO keeps Destination Port and the raw
    attack label, aligned row-for-row with X and y."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1).to_numpy()
    return {
        "X": X[valid].reset_index(drop=True),
        "port": pd.to_numeric(df["Destination Port"], errors="coerce").fillna(-1).to_numpy()[valid],
        "y": (df["binary_label"] == "ATTACK").astype(int).to_numpy()[valid],
        "label": df["Label"].astype(str).to_numpy()[valid],
    }


def _load_train_val(processed_dir):
    return (_load_aligned(load_split(processed_dir, "train")),
            _load_aligned(load_split(processed_dir, "val")))


def _at_fpr(scores, y, target_fpr):
    """Threshold from benign scores, strict `>` flagging. Returns the
    ACHIEVED fpr and recall, which can differ from target when scores tie."""
    thr = pick_threshold_at_fpr(scores[y == 0], target_fpr)
    flag = scores > thr
    return thr, float(flag[y == 0].mean()), float(flag[y == 1].mean())


def _train_rf(X, y, seed, n_estimators=200):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                  n_jobs=-1).fit(X, y)


def _rf_scores(Xtr, ytr, Xva, seed):
    rf = _train_rf(Xtr, ytr, seed)
    return rf, rf.predict_proba(Xva)[:, 1]


def _ae_scores(Xtr_benign, Xva, seed):
    """Train the autoencoder on scaled benign rows, return (model, scaler, val scores)."""
    from src.autoencoder import fit_scaler, train_autoencoder, reconstruction_error
    scaler = fit_scaler(Xtr_benign)
    model = train_autoencoder(scaler.transform(Xtr_benign), seed=seed)
    return model, scaler, reconstruction_error(model, scaler.transform(Xva))


def _summary_row(name, scores, y, target_fpr=0.01):
    thr, fpr, rec = _at_fpr(scores, y, target_fpr)
    return {"variant": name, "auroc": roc_auc_score(y, scores),
            "thr": thr, "achieved_fpr": fpr, f"recall@{target_fpr:.0%}": rec}


def _per_attack(labels, y, flag):
    out = []
    for lab in sorted(set(labels[y == 1])):
        m = labels == lab
        out.append((lab, int(m.sum()), float(flag[m].mean())))
    return out


# ------------------------------------------------------- 1. files & features

def list_inputs(raw_dir="data/raw", processed_dir="data/processed"):
    """Check 1: exact source files, retained features, split sizes."""
    from src.data_prep import _find_day_file, DROP_FROM_FEATURES
    raw_dir = Path(raw_dir)
    print("=== Source files ===")
    for day in ("Monday", "Wednesday"):
        try:
            print(f"  {day:9s}: {_find_day_file(raw_dir, day).name}")
        except FileNotFoundError as e:
            print(f"  {day:9s}: not found in {raw_dir} ({e})")

    train, val = _load_train_val(processed_dir)
    feats = list(train["X"].columns)
    print(f"\n=== Retained features: {len(feats)} ===")
    for i, c in enumerate(feats, 1):
        print(f"  {i:2d}. {c}")
    print(f"\n  Excluded from features: {DROP_FROM_FEATURES} (context) "
          f"+ duplicate 'Fwd Header Length.1' + label/bookkeeping columns")

    print("\n=== Rows after NaN/inf filtering ===")
    for name, d in (("train", train), ("val", val)):
        n_att = int(d["y"].sum())
        print(f"  {name:5s}: {len(d['y']):7d}  (benign {len(d['y']) - n_att}, attack {n_att})")
        print("         attack labels:", {k: int(v) for k, v in pd.Series(d["label"][d["y"] == 1]).value_counts().items()})
    return feats


# ---------------------------------------------------------- 2. port-only

def port_only_check(processed_dir="data/processed", seed=42):
    """Check 2a: Destination Port as the ONLY feature. Reports AUROC and
    recall at several FPRs, and tests whether 0% recall at 1% FPR is a tie
    artifact (benign mass sitting exactly on the threshold score)."""
    tr, va = _load_train_val(processed_dir)
    rf = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
    rf.fit(tr["port"].reshape(-1, 1), tr["y"])
    s = rf.predict_proba(va["port"].reshape(-1, 1))[:, 1]
    y = va["y"]

    print(f"=== Destination Port ONLY (RF, 50 trees) - AUROC {roc_auc_score(y, s):.4f} ===")
    print(f"  distinct score values on val: {len(np.unique(s))}")
    print(f"  {'target FPR':>10s} {'achieved FPR':>13s} {'recall':>8s} {'benign tied @thr':>17s}")
    for f in FPR_GRID:
        thr, fpr, rec = _at_fpr(s, y, f)
        tied = float((s[y == 0] == thr).mean())
        print(f"  {f:>10.0%} {fpr:>13.4f} {rec:>8.4f} {tied:>17.4f}")

    print("\n  Top destination ports by share of each class (val):")
    for name, mask in (("benign", y == 0), ("attack", y == 1)):
        shares = pd.Series(va["port"][mask]).value_counts(normalize=True).head(5)
        print(f"    {name}: " + ", ".join(f"{int(p)}={v:.1%}" for p, v in shares.items()))
    print("  If benign port 80 > 1% and attack port 80 is high, the 1% threshold sits")
    print("  on port 80's score, strict `>` flags nothing, and 0% recall is a tie artifact.")
    return s


# --------------------------------------------------------- 3. ablation

def context_ablation(processed_dir="data/processed", seeds=(42,), target_fpr=0.01,
                     include_ae=True):
    """Check 2b: both models with and without Destination Port as an input.
    The AE costs ~90 s per fit, so default is one seed; pass seeds=(42, 7, 123)
    for the full version (about 18 min for the AE alone)."""
    tr, va = _load_train_val(processed_dir)
    ytr, yva = tr["y"], va["y"]
    variants = {
        "without port (current)": (tr["X"].to_numpy(), va["X"].to_numpy()),
        "with port": (np.column_stack([tr["X"].to_numpy(), tr["port"]]),
                      np.column_stack([va["X"].to_numpy(), va["port"]])),
    }
    rows = []
    for seed in seeds:
        for vname, (Xtr, Xva) in variants.items():
            print(f"\n--- RF-IDS | {vname} | seed {seed} ---")
            _, s = _rf_scores(Xtr, ytr, Xva, seed)
            rows.append({"model": "RF-IDS", "seed": seed, **_summary_row(vname, s, yva, target_fpr)})
            if include_ae:
                print(f"\n--- Autoencoder | {vname} | seed {seed} ---")
                _, _, s = _ae_scores(Xtr[ytr == 0], Xva, seed)
                rows.append({"model": "Autoencoder", "seed": seed, **_summary_row(vname, s, yva, target_fpr)})
    df = pd.DataFrame(rows)
    print("\n=== Context ablation (val) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


# ------------------------------------------------------- 4. RF leakage

def rf_leakage_check(processed_dir="data/processed", seed=42, target_fpr=0.01,
                     drop_top=(1, 3, 5), n_probe=5000, n_ref=100000):
    """Check 3: why is RF recall 1.0000? Looks at importances, retrains
    without the top-k features, and measures how close val attack flows are
    to train attack flows (exact duplicates and nearest-neighbour distance)."""
    from sklearn.neighbors import NearestNeighbors
    tr, va = _load_train_val(processed_dir)
    Xtr, ytr, Xva, yva = tr["X"], tr["y"], va["X"], va["y"]
    cols = list(Xtr.columns)

    print("--- baseline RF (all features) ---")
    rf, s = _rf_scores(Xtr.to_numpy(), ytr, Xva.to_numpy(), seed)
    base = _summary_row("all features", s, yva, target_fpr)
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    print("\n=== Top 10 feature importances ===")
    print(imp.head(10).to_string(float_format=lambda v: f"{v:.4f}"))

    rows = [base]
    for k in drop_top:
        dropped = list(imp.index[:k])
        keep = [c for c in cols if c not in dropped]
        print(f"\n--- retrain without top {k}: {dropped} ---")
        _, s_k = _rf_scores(Xtr[keep].to_numpy(), ytr, Xva[keep].to_numpy(), seed)
        rows.append(_summary_row(f"drop top {k}", s_k, yva, target_fpr))
    print("\n=== RF with top features removed (val) ===")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("  Recall that stays ~1.0 after dropping the top features = signal is spread")
    print("  across many features (plausible DoS separability). A collapse = reliance on")
    print("  one or two features worth explaining.")

    # exact duplicates between val and train (same feature vector)
    h = lambda X: pd.util.hash_pandas_object(X, index=False)
    h_tr_att, h_tr_ben = set(h(Xtr[ytr == 1])), set(h(Xtr[ytr == 0]))
    h_va_att = h(Xva[yva == 1])
    print("\n=== Exact-duplicate check (val attack rows vs train) ===")
    print(f"  identical to a train ATTACK row : {h_va_att.isin(h_tr_att).mean():.2%}")
    print(f"  identical to a train BENIGN row : {h_va_att.isin(h_tr_ben).mean():.2%}")

    # nearest-neighbour distance in standardized space
    rng = np.random.default_rng(seed)
    scaler = StandardScaler().fit(Xtr.to_numpy())
    Ztr, Zva = scaler.transform(Xtr.to_numpy()), scaler.transform(Xva.to_numpy())

    def nn_dist(cls, label_arr):
        ref_idx = np.flatnonzero(ytr == cls)
        probe_idx = np.flatnonzero(yva == cls)
        ref_idx = rng.choice(ref_idx, min(n_ref, len(ref_idx)), replace=False)
        probe_idx = rng.choice(probe_idx, min(n_probe, len(probe_idx)), replace=False)
        nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1).fit(Ztr[ref_idx])
        d = nn.kneighbors(Zva[probe_idx])[0][:, 0]
        return d, label_arr[probe_idx]

    d_att, lab_att = nn_dist(1, va["label"])
    d_ben, _ = nn_dist(0, va["label"])
    q = lambda d: {f"p{p}": float(np.percentile(d, p)) for p in (5, 50, 95)}
    print("\n=== Nearest-neighbour distance to same-class TRAIN rows (standardized) ===")
    print("  val attack -> train attack :", {k: round(v, 3) for k, v in q(d_att).items()},
          f"| share < 0.01: {(d_att < 0.01).mean():.1%}")
    print("  val benign -> train benign :", {k: round(v, 3) for k, v in q(d_ben).items()},
          f"| share < 0.01: {(d_ben < 0.01).mean():.1%}")
    print("  median by attack type:")
    for lab in sorted(set(lab_att)):
        print(f"    {lab:18s} n={int((lab_att == lab).sum()):5d}  median {np.median(d_att[lab_att == lab]):.3f}")
    print("  Attack flows far closer to train attacks than benign flows are to train")
    print("  benign = near-duplicate/templated traffic (expected for DoS tools, but it")
    print("  means high recall may not transfer to a different capture).")
    return {"importances": imp, "ablation": pd.DataFrame(rows)}


# ------------------------------------------------------- 5. error analysis

def _top_feature_gaps(Z_a, Z_ref, cols, top_n):
    gap = np.median(Z_a, axis=0) - np.median(Z_ref, axis=0)
    order = np.argsort(-np.abs(gap))[:top_n]
    return [(cols[i], float(gap[i])) for i in order]


def error_analysis(processed_dir="data/processed", seed=42, target_fpr=0.01, top_n=6):
    """Check 4: AE misses per attack type (what do the missed flows look like?)
    and RF false positives (what do they look like?). Median gaps are in
    benign-train standard-deviation units."""
    tr, va = _load_train_val(processed_dir)
    cols = list(tr["X"].columns)
    Xtr, ytr, Xva, yva, lab = tr["X"].to_numpy(), tr["y"], va["X"].to_numpy(), va["y"], va["label"]
    sc = StandardScaler().fit(Xtr[ytr == 0])
    Zva = sc.transform(Xva)

    print("--- Autoencoder ---")
    _, _, s_ae = _ae_scores(Xtr[ytr == 0], Xva, seed)
    thr, fpr, _ = _at_fpr(s_ae, yva, target_fpr)
    flag = s_ae > thr
    print(f"\nAE threshold {thr:.4f}, achieved FPR {fpr:.4f}")
    print(f"{'attack':18s} {'n':>6s} {'recall':>7s} {'median score / thr':>19s}")
    for name, n, rec in _per_attack(lab, yva, flag):
        m = lab == name
        print(f"{name:18s} {n:6d} {rec:7.3f} {np.median(s_ae[m]) / thr:19.2f}")
    for name, n, rec in _per_attack(lab, yva, flag):
        m = lab == name
        missed, caught = m & ~flag, m & flag
        if missed.sum() == 0:
            continue
        ref, ref_name = (Zva[caught], "caught same-type flows") if caught.sum() >= 5 \
            else (Zva[yva == 0], "benign val flows")
        print(f"\n  {name}: {int(missed.sum())} missed. Largest median gaps vs {ref_name}:")
        for c, g in _top_feature_gaps(Zva[missed], ref, cols, top_n):
            print(f"    {c:35s} {g:+.2f} sd")

    print("\n--- RF-IDS ---")
    rf, s_rf = _rf_scores(Xtr, ytr, Xva, seed)
    thr, fpr, rec = _at_fpr(s_rf, yva, target_fpr)
    flag = s_rf > thr
    print(f"\nRF threshold {thr:.4f} (~{thr * rf.n_estimators:.0f} of {rf.n_estimators} trees), "
          f"achieved FPR {fpr:.4f}, recall {rec:.4f}")
    for name, n, r in _per_attack(lab, yva, flag):
        print(f"  {name:18s} n={n:6d} recall {r:.3f}")
    fp = (yva == 0) & flag
    print(f"\n  RF false positives: {int(fp.sum())} of {int((yva == 0).sum())} benign val flows")
    if fp.sum():
        votes = np.rint(s_rf[fp] * rf.n_estimators).astype(int)
        vc = pd.Series(votes).value_counts().sort_index().head(8)
        print("  tree votes for attack among false positives:", {int(k): int(v) for k, v in vc.items()})
        print(f"  Largest median gaps, false positives vs other benign:")
        ref = Zva[(yva == 0) & ~flag]
        for c, g in _top_feature_gaps(Zva[fp], ref, cols, top_n):
            print(f"    {c:35s} {g:+.2f} sd")
    return {"ae_scores": s_ae, "rf_scores": s_rf}


# ------------------------------------------------------------------ runner

def run_all(raw_dir="data/raw", processed_dir="data/processed", seeds=(42,)):
    """Everything except the long AE items is quick. Run pieces individually
    from the notebook if you only want one check."""
    list_inputs(raw_dir, processed_dir)
    port_only_check(processed_dir, seed=seeds[0])
    context_ablation(processed_dir, seeds=seeds)
    rf_leakage_check(processed_dir, seed=seeds[0])
    error_analysis(processed_dir, seed=seeds[0])


if __name__ == "__main__":
    run_all()
