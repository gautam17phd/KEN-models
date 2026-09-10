"""
Two-panel scatter/correlation plot comparing predicted vs reference magmoms
for CHGNet and KENJI (zero-mode), side by side.

Reads the raw per-atom pred/ref .npz files saved by chgnet_magmom_eval.py and
kenji_eval.py:
    chgnet_magmom_results_raw.npz
    kenji_eval_results_zero_raw.npz

Usage:
    python plot_magmom_correlation.py \
        --chgnet chgnet_magmom_results_raw.npz \
        --kenji kenji_eval_results_zero_raw.npz \
        --out magmom_correlation.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def compute_stats(pred, ref):
    diff = pred - ref
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    
    # Coefficient of Determination (R^2)
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    
    return mae, rmse, r2


def plot_panel(ax, pred, ref, title, use_abs=False):
    if use_abs:
        pred = np.abs(pred)
        ref = np.abs(ref)

    mae, rmse, r2 = compute_stats(pred, ref)

    # Modified scatter properties to use red circles with black boundaries
    ax.scatter(
        ref, pred, 
        s=12, 
        facecolors="red", 
        edgecolors="black", 
        linewidths=0.4, 
        alpha=0.4
    )

    lo = min(ref.min(), pred.min())
    hi = max(ref.max(), pred.max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("|Reference magmom| (μ$_B$)" if use_abs else "Reference magmom (μ$_B$)")
    ax.set_ylabel("|Predicted magmom| (μ$_B$)" if use_abs else "Predicted magmom (μ$_B$)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    # Modified text readout to display R^2 instead of R
    stats_text = f"MAE   = {mae:.3f} μ$_B$\nRMSE  = {rmse:.3f} μ$_B$\nR$^2$    = {r2:.3f}"
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend(loc="lower right", fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chgnet", type=str, default="chgnet_magmom_results_raw.npz")
    parser.add_argument("--kenji", type=str, default="kenji_eval_results_zero_raw.npz")
    parser.add_argument("--kenji-label", type=str, default=None,
                         help="Panel title for the KENJI subplot, e.g. 'KENJI (20% perturbed)'. "
                              "If omitted, derived from the --kenji filename.")
    parser.add_argument("--out", type=str, default="magmom_correlation.png")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--abs", action="store_true",
                         help="Plot |pred| vs |ref| instead of signed values "
                              "(useful when the model struggles with sign but "
                              "captures magnitude correctly)")
    args = parser.parse_args()

    chgnet_data = np.load(args.chgnet)
    kenji_data = np.load(args.kenji)

    kenji_label = args.kenji_label or f"KENJI ({Path(args.kenji).stem})"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    plot_panel(axes[0], chgnet_data["pred"], chgnet_data["ref"], "CHGNet", use_abs=args.abs)
    plot_panel(axes[1], kenji_data["pred"], kenji_data["ref"], kenji_label, use_abs=args.abs)

    title = "Predicted vs reference |magnetic moment|" if args.abs else "Predicted vs reference magnetic moments"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()

