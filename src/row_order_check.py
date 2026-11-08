"""
Row-order check: is the row order of a MachineLearningCSV day file the
chronological order of the capture?

Why: our split is positional per label group (first 70% of each label's rows
-> train, next 15% -> val, last 15% -> test). Calling that "time-ordered" is
only true if row order tracks capture time WITHIN each label group.
MachineLearningCSV has no timestamps, so we borrow them from the
labelled-flows release (the TrafficLabelling folder).

Notes on the labelled-flows Timestamp column:
  - It is a 12-hour clock with no AM/PM (afternoon shows as 1:00-5:00), so
    hours below `pm_if_hour_below` are shifted +12. Default 8 fits every
    CICIDS2017 day (captures start at 8-9 AM).
  - Resolution is one minute, so ties are common; "non-decreasing" counts a
    tie as in order.
  - Dates may print as May instead of July (day/month swapped). That does
    not affect ordering within a single day file.

Reads raw CSVs only. Never touches data/processed or the test split.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ALIGN_COLS = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets"]


def _read(path):
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    df.columns = df.columns.str.strip()
    return df


def _order_stats(secs):
    """secs: times in row order. Returns (spearman, share non-decreasing, #drops > 5 min)."""
    step = np.diff(secs)
    rho = pd.Series(secs).corr(pd.Series(np.arange(len(secs))), method="spearman")
    return float(rho), float((step >= 0).mean()), int((step < -300).sum())


def check_row_order(ml_csv, labelled_csv, pm_if_hour_below=8):
    """ml_csv: e.g. data/raw/Wednesday-workingHours.pcap_ISCX.csv
    labelled_csv: the same day from the labelled-flows release (has Timestamp)."""
    if Path(ml_csv).resolve() == Path(labelled_csv).resolve():
        print("Both paths point to the SAME file. The second path must be the Wednesday CSV")
        print("from the labelled-flows download (TrafficLabelling), not the one in data/raw.")
        return None
    ml, lab = _read(ml_csv), _read(labelled_csv)
    print(f"MachineLearningCSV rows: {len(ml):,}   labelled-flows rows: {len(lab):,}")
    if "Timestamp" not in lab.columns:
        print("The second file has no 'Timestamp' column, so it is not the labelled-flows file.")
        print("Columns it does have (first 12):", list(lab.columns[:12]))
        return None
    if len(ml) != len(lab):
        print("Row counts differ, so positional alignment is NOT guaranteed.")
        print("Stop here and tell Claude the two counts; a key-based merge is needed.")
        return None

    # --- step 1: row-for-row alignment on shared columns
    print("\nAlignment (share of rows where the shared column matches exactly):")
    all_ok = True
    for c in ALIGN_COLS:
        if c not in ml.columns or c not in lab.columns:
            print(f"  {c:25s} missing in one file, skipped")
            continue
        a = pd.to_numeric(ml[c], errors="coerce").to_numpy()
        b = pd.to_numeric(lab[c], errors="coerce").to_numpy()
        same = np.isclose(a, b, equal_nan=True)
        print(f"  {c:25s} {same.mean():.4%}")
        all_ok &= same.mean() > 0.999
    print("  ->", "files align row-for-row" if all_ok else
          "files do NOT align row-for-row; do not trust timestamps by position")
    if not all_ok:
        return None

    # --- step 2: time order, after undoing the 12-hour clock
    ts = pd.to_datetime(lab["Timestamp"], errors="coerce")
    ok = ts.notna().to_numpy()
    print(f"\nTimestamps parsed: {ok.mean():.2%}")
    ts = ts + pd.to_timedelta(np.where(ts.dt.hour < pm_if_hour_below, 12, 0), unit="h")
    secs = (ts - ts.min()).dt.total_seconds().to_numpy()
    print(f"Time range after 12-hour fix: {ts.min().strftime('%H:%M')} -> {ts.max().strftime('%H:%M')}")

    rho, nd, drops = _order_stats(secs[ok])
    print("\nWHOLE FILE, row order vs time:")
    print(f"  Spearman(row index, time)       : {rho:.4f}")
    print(f"  consecutive pairs non-decreasing: {nd:.2%}")
    print(f"  backward jumps larger than 5 min: {drops:,}")

    # --- the view that matters: our split cuts each label group by position
    if "Label" in lab.columns:
        print("\nPER LABEL (our split cuts each label group by row position):")
        print(f"  {'label':18s} {'rows':>8s} {'Spearman':>9s} {'non-decr.':>10s} {'drops>5min':>11s}  time span")
        labels = lab["Label"].astype(str).str.strip().to_numpy()
        for name in pd.Series(labels).value_counts().index:
            m = (labels == name) & ok
            if m.sum() < 3:
                print(f"  {name:18s} {m.sum():8d}  (too few rows)")
                continue
            r, n_d, d = _order_stats(secs[m])
            t0 = ts[m].min().strftime("%H:%M"); t1 = ts[m].max().strftime("%H:%M")
            print(f"  {name:18s} {m.sum():8d} {r:9.3f} {n_d:10.2%} {d:11,d}  {t0}-{t1}")

    # --- how are rows laid out? time span per tenth of the file
    print("\nTime span of each tenth of the file, in row order:")
    edges = np.linspace(0, len(secs), 11, dtype=int)
    for i in range(10):
        seg = ts.iloc[edges[i]:edges[i + 1]]
        print(f"  rows {edges[i]:>7,}-{edges[i+1]:>7,}: {seg.min().strftime('%H:%M')} -> {seg.max().strftime('%H:%M')}")

    print("\nReading it: Spearman ~1 and ~100% non-decreasing (also within each label) means")
    print("row order IS chronological, so the positional split is a within-day temporal split.")
    print("Low values mean it is a 'position-based' split and related flows may sit in")
    print("different partitions. If every tenth of the file spans the whole day, rows are")
    print("effectively shuffled with respect to time.")
    return {"spearman": rho, "nondecreasing": nd, "drops_gt_5min": drops}


if __name__ == "__main__":
    import sys
    check_row_order(sys.argv[1], sys.argv[2])
