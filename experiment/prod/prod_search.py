"""Fig. 9 from a search run's saved output, into results/figures/<hash>/.

    uv run python experiment/prod/prod_search.py <search hash>                 # log-T, auto range
    uv run python experiment/prod/prod_search.py <search hash> --paper-range   # linear T in [1, 20]
"""

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lab.data.dataset import NOISY_VARIANTS, VARIANTS
from lab.paths import FIGURES_DIR, PROJECT_ROOT
from lab.runner import load_run


def figure_dir(run_id):
    path = FIGURES_DIR / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def sweep(true_losses, false_losses, thresholds):
    """FN and FP (%) per threshold, accepting a pair when L* <= T.

    FN(T): true pairs with L* > T.  FP(T): false pairs with L* <= T.
    """
    true_losses = np.asarray(true_losses, dtype=float)
    false_losses = np.asarray(false_losses, dtype=float)
    fn = np.array([100.0 * np.mean(true_losses > t) for t in thresholds])
    fp = np.array([100.0 * np.mean(false_losses <= t) for t in thresholds])
    return fn, fp


def figure9(args):
    """Siminski et al. Fig. 9, boundary-value method, 0.7" case: FN and FP vs T.

    --paper-range uses the paper's linear 1-20 axis; true pairs the search
    misses sit far above 20, so FN plateaus above zero there. The default
    log axis spans the actual L* range.
    """
    output_dir = load_run(args.hash, "search")
    with np.load(output_dir / f"association_{args.variant}.npz") as archive:
        is_true, loss = archive["is_true"], archive["loss"]

    true_losses, false_losses = loss[is_true], loss[~is_true]
    if not true_losses.size or not false_losses.size:
        raise SystemExit("Fig. 9 needs both true and false pairs; "
                         "generate at least two objects")

    print(f"  {len(true_losses)} true pairs   L* in "
          f"[{min(true_losses):.2e}, {max(true_losses):.2e}]")
    print(f"  {len(false_losses)} false pairs  L* in "
          f"[{min(false_losses):.2e}, {max(false_losses):.2e}]")

    if args.paper_range:
        # L* is roughly chi-squared(4), so 1-20 is its useful band.
        thresholds = np.linspace(1.0, 20.0, 400)
        xscale = "linear"
    else:
        # Log-spaced over the actual L*: false pairs reach ~1e6.
        lo = min(min(true_losses), min(false_losses))
        hi = max(max(true_losses), max(false_losses))
        thresholds = np.logspace(np.log10(lo), np.log10(hi), 500)
        xscale = "log"

    fn, fp = sweep(true_losses, false_losses, thresholds)

    # FN = FP = 0 between the worst true and best false L*, if they separate.
    gap_lo, gap_hi = max(true_losses), min(false_losses)
    if gap_lo < gap_hi:
        print(f"  FN = FP = 0 for T in ({gap_lo:.3e}, {gap_hi:.3e})")
    else:
        print("  no threshold gives FN = FP = 0 "
              f"(worst true L* {gap_lo:.2e} >= best false L* {gap_hi:.2e})")

    # Twin y-axes, as in the paper: FN and FP have very different scales.
    fig, ax_fn = plt.subplots(figsize=(7.0, 4.5))
    ax_fp = ax_fn.twinx()

    ax_fn.set_xscale(xscale)
    l1, = ax_fn.plot(thresholds, fn, color="#1f1f1f", marker="^", markevery=0.06,
                     markersize=5, lw=1.4, label="False negative (boundary-value)")
    l2, = ax_fp.plot(thresholds, fp, color="#8c8c8c", marker="D", markevery=0.06,
                     markersize=4, lw=1.4, label="False positive (boundary-value)")

    # Shade the clean-separation window, if there is one.
    if gap_lo < gap_hi:
        ax_fn.axvspan(gap_lo, gap_hi, color="#cfe8cf", alpha=0.5, zorder=0)

    ax_fn.set_xlabel("Decision threshold  T   (accept pair when L* $\\leq$ T)")
    ax_fn.set_ylabel("False negative [%]")
    ax_fp.set_ylabel("False positive [%]")
    ax_fn.set_ylim(0, max(5, fn.max() * 1.1))
    ax_fp.set_ylim(0, max(5, fp.max() * 1.1))
    sigma = NOISY_VARIANTS.get(args.variant, 0.0)
    ax_fn.set_title(f"Fig. 9 analogue -- {args.variant} chain "
                    f"($\\sigma_{{\\alpha,\\delta}} = {sigma}''$), boundary-value only")
    ax_fn.grid(True, alpha=0.25)
    ax_fn.legend(handles=[l1, l2], loc="center right", framealpha=0.9)

    fig.tight_layout()
    out = figure_dir(args.hash) / (
        f"figure9_fp_fn_{args.variant}"
        f"{'_paper-range' if args.paper_range else ''}.png"
    )
    fig.savefig(out, dpi=150)
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw Fig. 9 from a search run.")
    parser.add_argument("hash", help="hash of a search run")
    parser.add_argument(
        "--paper-range", action="store_true",
        help="linear threshold axis T in [1, 20], matching the paper exactly",
    )
    parser.add_argument(
        "--variant", default="noisy_0.7as", choices=VARIANTS,
        help="dataset chain (default: noisy_0.7as, the paper's 0.7\" case)",
    )
    figure9(parser.parse_args())
