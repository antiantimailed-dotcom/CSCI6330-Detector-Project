"""
Shared data preparation for the Baseline (autoencoder) vs RF-IDS comparison.
Both detectors must read from the same train/val/test files produced here —
do not re-split the data separately for each model.
"""

from pathlib import Path
import pandas as pd
import numpy as np

SEED = 42
RNG = np.random.default_rng(SEED)

# Columns we don't want feeding either model. CICIDS2017's MachineLearningCSV
# files don't include IPs/timestamps, but Destination Port is a known shortcut
# feature (e.g. attacks clustering on specific ports), so we keep it out of X
# by default and store it separately in case you want to inspect it later.
DROP_FROM_FEATURES = ["Destination Port"]


def load_and_clean(path: Path, day_name: str) -> pd.DataFrame:
    """Load one day's CSV, strip whitespace from column names and labels,
    tag rows with their source day, and add a binary label."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    df["Label"] = df["Label"].str.strip()
    df["day"] = day_name
    df["binary_label"] = np.where(df["Label"] == "BENIGN", "BENIGN", "ATTACK")

    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate flows. CICIDS2017 is known to contain near-identical
    repeated flows that inflate scores if split randomly across train/test."""
    feature_cols = [c for c in df.columns if c not in ("day",)]
    before = len(df)
    df = df.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)
    print(f"  dropped {before - len(df)} duplicate rows ({before} -> {len(df)})")
    return df


def _stratified_three_way_split(df: pd.DataFrame, strat_col: str,
                                 train_frac=0.7, val_frac=0.15):
    """Split a dataframe into train/val/test, preserving the proportion of
    each value in strat_col in all three parts. test_frac = 1 - train - val.

    Ordered by ROW POSITION within each strat_col group, not shuffled. This
    file's source CSV is written out in roughly the order flows occurred, so
    position is a proxy for time: train gets the earliest rows, val the next
    slice, test the latest. This is NOT a full day-based split (that needs
    more attack days than this pilot has — see build_pilot_splits), but it's
    a meaningful step up from a random shuffle, which could otherwise place
    two near-identical, millisecond-apart flows from the same burst into
    both train and test and let the model effectively see its own test
    data. Positional splitting keeps temporally-close flows together on one
    side of the split instead."""
    train_parts, val_parts, test_parts = [], [], []
    for _, group in df.groupby(strat_col):
        n = len(group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])
    # Shuffling the FINAL combined splits (across label groups) is fine —
    # it only affects row order for training, not which split a row lands
    # in, so it doesn't undo the positional logic above.
    return (pd.concat(train_parts).sample(frac=1, random_state=SEED).reset_index(drop=True),
            pd.concat(val_parts).sample(frac=1, random_state=SEED).reset_index(drop=True),
            pd.concat(test_parts).sample(frac=1, random_state=SEED).reset_index(drop=True))


def build_pilot_splits(monday: pd.DataFrame, wednesday: pd.DataFrame):
    """
    Pilot split strategy (milestone 1 scope — revisit in milestone 2 once more
    days are added):
      - Monday (benign-only day) is split 70/15/15 and supplies benign rows
        to all three partitions.
      - Wednesday (DoS/DDoS/Heartbleed + benign) is split 70/15/15 stratified
        by the specific attack label, so every attack subtype is represented
        in train, val, and test.
      - train:  used to fit both the autoencoder (benign rows only) and
                RF-IDS (benign + attack rows).
      - val:    used ONLY to pick the autoencoder's anomaly threshold (at a
                target false-positive rate) and to tune RF-IDS hyperparameters.
      - test:   touched exactly once, at the end, for the final comparison.
    This is a same-day split, not a cross-day split — a real limitation to
    flag in your report. Within each day, rows are split by POSITION (a proxy
    for capture time), not randomly, so near-duplicate back-to-back flows
    from the same burst land on one side of the split rather than being
    randomly scattered across train/val/test. True day-based generalization
    (e.g. training only on Wednesday's attacks and testing on Friday's,
    once more attack days exist) is milestone 2 work — this pilot only has
    one attack day, so there's nothing yet to hold a day out against.
    """
    mon_train, mon_val, mon_test = _stratified_three_way_split(monday, "binary_label")
    wed_train, wed_val, wed_test = _stratified_three_way_split(wednesday, "Label")

    train = pd.concat([mon_train, wed_train]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    val = pd.concat([mon_val, wed_val]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    test = pd.concat([mon_test, wed_test]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    return train, val, test


def _find_day_file(raw_dir: Path, day_prefix: str) -> Path:
    """Locate a day's CSV by prefix instead of an exact filename — CICIDS2017
    mirrors vary in using '.' or '_' before 'pcap' (e.g. Monday-WorkingHours.pcap_ISCX.csv
    vs Monday-WorkingHours_pcap_ISCX.csv), and this avoids breaking on that."""
    matches = sorted(raw_dir.glob(f"{day_prefix}-*.csv"))
    if not matches:
        available = sorted(p.name for p in raw_dir.glob("*.csv"))
        raise FileNotFoundError(
            f"No CSV starting with '{day_prefix}-' found in {raw_dir}.\n"
            f"Files actually present there: {available or '(none)'}"
        )
    if len(matches) > 1:
        print(f"  warning: multiple matches for '{day_prefix}-*.csv', using {matches[0].name}: {[m.name for m in matches]}")
    return matches[0]


def run_pilot_split(raw_dir: str = "data/raw", out_dir: str = "data/processed"):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Monday (benign)...")
    monday = load_and_clean(_find_day_file(raw_dir, "Monday"), "Monday")
    print("Loading Wednesday (DoS/DDoS + Heartbleed + benign)...")
    wednesday = load_and_clean(_find_day_file(raw_dir, "Wednesday"), "Wednesday")

    print("Deduplicating...")
    monday = dedupe(monday)
    wednesday = dedupe(wednesday)

    print("Splitting...")
    train, val, test = build_pilot_splits(monday, wednesday)

    print(f"\nFinal sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name} label counts:\n{split['binary_label'].value_counts().to_string()}")

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(f"\nSaved to {out_dir}/{{train,val,test}}.csv")

    return train, val, test


if __name__ == "__main__":
    run_pilot_split()
