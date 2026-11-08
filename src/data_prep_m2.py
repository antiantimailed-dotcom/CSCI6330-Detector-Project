"""
Milestone 2 data preparation (cross-day, unseen attack families).

Design
  SEEN days  (Monday, Tuesday, Wednesday): split by row position within each
             (day, Label) group, 70% train / 15% val / 15% test_seen.
             Same rule as Milestone 1, extended with Tuesday.
  UNSEEN days (Thursday AM web attacks, Thursday PM infiltration, Friday AM
             bot, Friday PM port scan, Friday PM DDoS): held out WHOLE as
             test_unseen. No row from these days ever enters train or val,
             so every attack family in test_unseen is new to both detectors
             and the benign traffic in it comes from different days.

What is and isn't "temporal"
  Milestone 1 row-order check (Wednesday): attack rows are in capture-time
  order within each attack type, benign rows are NOT (Spearman 0.42-0.45).
  So the positional split is chronological for attacks and arbitrary for
  benign. Tuesday has not been checked yet: run row_order_check on it before
  describing Tuesday's attack split as chronological.

Does not touch Milestone 1 files: reads data/raw, writes data/processed_m2/.
Never scores anything.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import SEED, dedupe

# key (written to the `day` column) -> filename prefix, matched case-insensitively
SEEN_FILES = {
    "Monday": "Monday-WorkingHours",
    "Tuesday": "Tuesday-WorkingHours",
    "Wednesday": "Wednesday-workingHours",
}
UNSEEN_FILES = {
    "Thu-AM-WebAttacks": "Thursday-WorkingHours-Morning-WebAttacks",
    "Thu-PM-Infiltration": "Thursday-WorkingHours-Afternoon-Infilteration",
    "Fri-AM-Bot": "Friday-WorkingHours-Morning",
    "Fri-PM-PortScan": "Friday-WorkingHours-Afternoon-PortScan",
    "Fri-PM-DDoS": "Friday-WorkingHours-Afternoon-DDos",
}


def _find_file(raw_dir: Path, prefix: str):
    matches = sorted(p for p in raw_dir.glob("*.csv") if p.name.lower().startswith(prefix.lower()))
    return matches[0] if matches else None


def check_files(raw_dir="data/raw"):
    """Print which of the 8 expected day files are present. Returns missing keys."""
    raw_dir = Path(raw_dir)
    missing = []
    print(f"Looking in {raw_dir.resolve()}")
    for group, files in (("seen", SEEN_FILES), ("unseen", UNSEEN_FILES)):
        for key, prefix in files.items():
            p = _find_file(raw_dir, prefix)
            print(f"  [{group:6s}] {key:20s} {p.name if p else 'MISSING (expected ' + prefix + '*.csv)'}")
            if p is None:
                missing.append(key)
    return missing


def _clean_label(s: str) -> str:
    """Strip whitespace and replace any non-ASCII junk (the web-attack labels
    contain a stray cp1252 dash) with ' - ', e.g. 'Web Attack - Brute Force'."""
    return re.sub(r"\s+", " ", re.sub(r"[^\x00-\x7f]+", " - ", s.strip())).replace(" - - ", " - ").strip()


def _load_day(path: Path, day_key: str) -> pd.DataFrame:
    """Same columns as data_prep.load_and_clean, but latin-1 (the web-attack
    file is not valid UTF-8) and with label cleaning."""
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).map(_clean_label)
    df["day"] = day_key
    df["binary_label"] = np.where(df["Label"] == "BENIGN", "BENIGN", "ATTACK")
    return df


def _split_positional(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    """Per (day, Label) group, by row position (groupby keeps file order)."""
    tr, va, te = [], [], []
    for _, g in df.groupby(["day", "Label"], sort=True):
        n = len(g)
        a = int(n * train_frac)
        b = a + int(n * val_frac)
        tr.append(g.iloc[:a]); va.append(g.iloc[a:b]); te.append(g.iloc[b:])
    shuf = lambda parts: pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return shuf(tr), shuf(va), shuf(te)


def _counts(df):
    return {f"{d} | {l}": int(n) for (d, l), n in df.groupby(["day", "Label"]).size().items()}


def run_m2_split(raw_dir="data/raw", out_dir="data/processed_m2"):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    missing = check_files(raw_dir)
    if missing:
        raise FileNotFoundError(f"Missing day files: {missing}. Copy them from the MachineLearningCSV "
                                f"zip into {raw_dir} and rerun.")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading and de-duplicating seen days...")
    seen = []
    for key, prefix in SEEN_FILES.items():
        print(f" {key}")
        seen.append(dedupe(_load_day(_find_file(raw_dir, prefix), key)))
    seen = pd.concat(seen, ignore_index=True)

    print("\nLoading and de-duplicating unseen days...")
    unseen = []
    for key, prefix in UNSEEN_FILES.items():
        print(f" {key}")
        unseen.append(dedupe(_load_day(_find_file(raw_dir, prefix), key)))
    unseen = pd.concat(unseen, ignore_index=True)

    # --- safety checks: unseen attack families must not appear in train data
    seen_attacks = set(seen.loc[seen.binary_label == "ATTACK", "Label"])
    unseen_attacks = set(unseen.loc[unseen.binary_label == "ATTACK", "Label"])
    overlap = seen_attacks & unseen_attacks
    print(f"\nSeen attack families  : {sorted(seen_attacks)}")
    print(f"Unseen attack families: {sorted(unseen_attacks)}")
    if overlap:
        raise ValueError(f"Attack labels appear in both seen and unseen days: {sorted(overlap)}. "
                         f"They would not be unseen families.")

    train, val, test_seen = _split_positional(seen)
    unseen = unseen.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # --- report exact feature-vector overlap between unseen test and train
    feat_cols = [c for c in train.columns if c not in ("Label", "binary_label", "day", "Destination Port")]
    h_tr = set(pd.util.hash_pandas_object(train[feat_cols], index=False))
    h_un = pd.util.hash_pandas_object(unseen[feat_cols], index=False)
    ov = h_un.isin(h_tr)
    print("\nExact feature-vector overlap, unseen rows that also appear in train (reported, not removed):")
    for key in UNSEEN_FILES:
        m = (unseen["day"] == key).to_numpy()
        print(f"  {key:20s} {ov[m].mean():7.2%} of {int(m.sum()):,} rows")

    for name, d in (("train", train), ("val", val), ("test_seen", test_seen), ("test_unseen", unseen)):
        d.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"\n{name}: {len(d):,} rows (benign {int((d.binary_label == 'BENIGN').sum()):,}, "
              f"attack {int((d.binary_label == 'ATTACK').sum()):,})")
        print(d.groupby(["day", "Label"]).size().to_string())

    manifest = {
        "seed": SEED,
        "seen_files": {k: _find_file(raw_dir, p).name for k, p in SEEN_FILES.items()},
        "unseen_files": {k: _find_file(raw_dir, p).name for k, p in UNSEEN_FILES.items()},
        "rule": "seen days: positional 70/15/15 per (day, Label); unseen days held out whole",
        "rows": {n: len(d) for n, d in (("train", train), ("val", val),
                                         ("test_seen", test_seen), ("test_unseen", unseen))},
        "counts": {n: _counts(d) for n, d in (("train", train), ("val", val),
                                               ("test_seen", test_seen), ("test_unseen", unseen))},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved train/val/test_seen/test_unseen + manifest.json to {out_dir}/")
    return train, val, test_seen, unseen


if __name__ == "__main__":
    run_m2_split()
