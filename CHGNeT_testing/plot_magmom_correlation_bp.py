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

import numpy as np
import matplotlib.pyplot as plt


def compute_stats(pred, ref):
    diff = pred - ref
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    # Pearson correlation coefficient
    r = np.corrcoef(pred, ref)[0, 1]
    return mae, rmse, r


def plot_panel(ax, pred, ref, title):
    mae, rmse, r = compute_stats(pred, ref)

    ax.scatter(ref, pred, s=8, alpha=0.35, edgecolors="none")

    lo = min(ref.min(), pred.min())
    hi = max(ref.max(), pred.max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Reference magmom (μ$_B$)")
    ax.set_ylabel("Predicted magmom (μ$_B$)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    stats_text = f"MAE  = {mae:.3f} μ$_B$\nRMSE = {rmse:.3f} μ$_B$\nR    = {r:.3f}"
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
    parser.add_argument("--out", type=str, default="magmom_correlation.png")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    chgnet_data = np.load(args.chgnet)
    kenji_data = np.load(args.kenji)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    plot_panel(axes[0], chgnet_data["pred"], chgnet_data["ref"], "CHGNet")
    plot_panel(axes[1], kenji_data["pred"], kenji_data["ref"], "KENJI (zero-mode)")

    fig.suptitle("Predicted vs reference magnetic moments", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
