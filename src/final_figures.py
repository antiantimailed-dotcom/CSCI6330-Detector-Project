import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MODELS_DIR = Path("models_m2")
OUT_DIR = Path("figures")
TARGET_FPR = 0.01

DETECTORS = {"Autoencoder": "score_autoencoder", "RF-IDS": "score_rf_ids"}
NAMES = {"Autoencoder": "Autoencoder", "RF-IDS": "Random forest"}

THEMES = {
    "dark": {"bg": "#1E1F22", "text": "#F2F2F2", "muted": "#A0A3A8", "grid": "#34363A",
             "AE": "#e76624", "RF": "#338ced", "cm_low": "#26282C", "cm_high": "#338ced", "flip": 0.0},
    "light": {"bg": "#FFFFFF", "text": "#1F1F1F", "muted": "#5F6368", "grid": "#E3E5E8",
              "AE": "#e76624", "RF": "#257de0", "cm_low": "#F2F6FB", "cm_high": "#257de0", "flip": 0.8},
}

FAMILIES = [("Web Attack - Brute Force", "Web brute\nforce"), ("Web Attack - XSS", "Web XSS"),
            ("DDoS", "DDoS"), ("PortScan", "PortScan"), ("Bot", "Bot")]
DAYS = [("Thu-AM-WebAttacks", "Thu AM\n(web)"), ("Thu-PM-Infiltration", "Thu PM\n(infiltration)"),
        ("Fri-AM-Bot", "Fri AM\n(bot)"), ("Fri-PM-PortScan", "Fri PM\n(portscan)"),
        ("Fri-PM-DDoS", "Fri PM\n(DDoS)")]


# ------------------------------------------------------- confusion matrices

def confusion_matrices(models_dir=MODELS_DIR, target_fpr=TARGET_FPR):
    """TP/FN/FP/TN per detector and split at the frozen threshold (strict >)."""
    thr = json.loads((Path(models_dir) / "thresholds.json").read_text())
    rows = []
    for split in ("seen", "unseen"):
        df = pd.read_csv(Path(models_dir) / f"test_scores_{split}.csv.gz")
        is_attack = df["y"].to_numpy() == 1
        for name, col in DETECTORS.items():
            flagged = df[col].to_numpy() > thr[name][str(target_fpr)]
            tp = int((flagged & is_attack).sum()); fn = int((~flagged & is_attack).sum())
            fp = int((flagged & ~is_attack).sum()); tn = int((~flagged & ~is_attack).sum())
            recall = tp / (tp + fn); precision = tp / (tp + fp)
            rows.append(dict(split=split, detector=name, TP=tp, FN=fn, FP=fp, TN=tn,
                             recall=recall, fpr=fp / (fp + tn), precision=precision,
                             f1=2 * precision * recall / (precision + recall),
                             accuracy=(tp + tn) / len(df)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ helpers

def _setup(t):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": t["text"],
                         "axes.labelcolor": t["muted"], "xtick.color": t["text"], "ytick.color": t["muted"]})


def _style_bars(ax, t):
    ax.set_facecolor(t["bg"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["muted"])
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)


def _grouped_bars(t, results, scope, items, path, ylim, fmt, xlabel, ylabel, ref=None):
    sub = results[(results.scope == scope) & (results.target_fpr == TARGET_FPR)]
    keys = [k for k, _ in items]
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    _style_bars(ax, t)
    x = np.arange(len(keys)); w = 0.34; gap = 0.02
    for offset, name, color in ((-w / 2 - gap, "Autoencoder", t["AE"]), (w / 2 + gap, "RF-IDS", t["RF"])):
        vals = [sub[(sub.item == k) & (sub.detector == name)].value.iloc[0] for k in keys]
        bars = ax.bar(x + offset, vals, w, color=color, label=NAMES[name], zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + ylim * 0.015, fmt(v),
                    ha="center", va="bottom", fontsize=11, color=t["text"])
    if ref is not None:
        ax.axhline(ref, color=t["muted"], linestyle=(0, (4, 3)), linewidth=1.3, zorder=2,
                   label=f"{ref:.0%} target")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in items])
    ax.set_ylim(0, ylim)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    legend = ax.legend(frameon=False, loc="upper left" if ref is not None else "upper right",
                       ncol=3, fontsize=12, handlelength=1.4)
    for txt in legend.get_texts():
        txt.set_color(t["text"])
    fig.tight_layout()
    fig.savefig(path, facecolor=t["bg"])
    plt.close(fig)


def _confusion_figure(t, cm, split, path):
    """Two 2x2 matrices side by side. Cell shade = share of that row (so the
    rare attack row is readable next to the huge normal row)."""
    cmap = LinearSegmentedColormap.from_list("seq", [t["cm_low"], t["cm_high"]])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    for ax, name in zip(axes, ("Autoencoder", "RF-IDS")):
        r = cm[(cm.split == split) & (cm.detector == name)].iloc[0]
        counts = np.array([[r.TP, r.FN], [r.FP, r.TN]], dtype=float)
        share = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                strong = share[i, j] > t["flip"]
                color = "#FFFFFF" if strong else t["text"]
                ax.text(j, i - 0.12, f"{int(counts[i, j]):,}", ha="center", va="center",
                        fontsize=14, fontweight="bold", color=color)
                ax.text(j, i + 0.18, f"{share[i, j]:.1%} of row", ha="center", va="center",
                        fontsize=10.5, color=color)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Flagged\nattack", "Passed as\nnormal"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Real\nattack", "Real\nnormal"])
        ax.tick_params(length=0, colors=t["text"])
        ax.set_xlabel("Model decision", labelpad=8)
        ax.set_ylabel("True label", labelpad=8)
        ax.set_title(NAMES[name], color=t["text"], fontsize=14, fontweight="bold", pad=10)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xticks([0.5], minor=True); ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color=t["bg"], linewidth=3)
        ax.tick_params(which="minor", length=0)
    fig.tight_layout(w_pad=3)
    fig.savefig(path, facecolor=t["bg"])
    plt.close(fig)


# --------------------------------------------------------------------- main

def make_figures(models_dir=MODELS_DIR, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    results = pd.read_csv(Path(models_dir) / "test_results_unseen.csv")
    cm = confusion_matrices(models_dir)

    for theme, t in THEMES.items():
        d = out_dir / theme
        d.mkdir(parents=True, exist_ok=True)
        _setup(t)
        _grouped_bars(t, results, "recall", FAMILIES, d / "recall_by_family.png", 1.12,
                      lambda v: f"{v:.0%}" if v >= 0.005 else "~0%",
                      "Unseen attack family", "Recall (share of attacks caught)")
        _grouped_bars(t, results, "fpr", DAYS, d / "fpr_by_day.png", 0.30, lambda v: f"{v:.1%}",
                      "Unseen day", "False-positive rate", ref=TARGET_FPR)
        _confusion_figure(t, cm, "unseen", d / "confusion_unseen.png")
        _confusion_figure(t, cm, "seen", d / "confusion_seen.png")

    cm.round(4).to_csv(out_dir / "confusion_matrices.csv", index=False)
    print(cm.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved figures to {out_dir}/dark (slides) and {out_dir}/light (report), "
          f"plus {out_dir}/confusion_matrices.csv")
    return cm


if __name__ == "__main__":
    make_figures()