"""Reproduce Siminski et al. Fig. 9 (the sigma = 0.7" case) for this pipeline.

Fig. 9 plots, against a decision threshold T applied to each tracklet
pair's optimised loss min L (= L*):

    false negative (FN) rate = true pairs rejected  (L* > T)   -- accept if L* <= T
    false positive (FP) rate = false pairs accepted (L* <= T)

as T is swept. The paper shows both the initial-value and boundary-value
formulations at three noise levels; this reimplementation only has the
boundary-value method, and its noisy chain is exactly the paper's 0.7"
case (0.7" relative + 5" systematic astrometric error), so one panel with
two curves is the faithful analogue.

    uv run python figures/figure9_fp_fn.py                 # log-T, auto range
    uv run python figures/figure9_fp_fn.py --paper-range   # linear T in [1, 20]

--paper-range matches the paper's axis exactly. With the optimiser in its
current state some true pairs do not converge, so L* for those stays far
above 20 and FN plateaus well above zero -- that gap IS the finding, not a
plotting artefact. The auto (log) range shows where true and false losses
actually separate.

Writes figures/figure9_fp_fn_<variant>[_paper-range].png.
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Pipeline stages import from sibling directories, not a package, so add
# them to the path before importing (same pattern as the other scripts).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("optimisation", "orbit", "hypothesis", "modelled", "discriminator"):
    sys.path.insert(0, str(_PROJECT_ROOT / _sub))

from optimise_hypothesis import run_association_test  # noqa: E402

_FIG_DIR = Path(__file__).resolve().parent


def sweep(true_losses, false_losses, thresholds):
    """FN and FP rate (percent) at each decision threshold.

    The gate accepts a pair when its optimised loss L* <= T, so:
      FN(T) = share of TRUE  pairs with L* >  T  (a real match rejected)
      FP(T) = share of FALSE pairs with L* <= T  (an unrelated pair accepted)
    Raising T lowers FN and raises FP; the paper picks T in the window that
    keeps FP low while still accepting most true pairs.
    """
    true_losses = np.asarray(true_losses, dtype=float)
    false_losses = np.asarray(false_losses, dtype=float)
    fn = np.array([100.0 * np.mean(true_losses > t) for t in thresholds])
    fp = np.array([100.0 * np.mean(false_losses <= t) for t in thresholds])
    return fn, fp


def main():
    parser = argparse.ArgumentParser(description="Reproduce Siminski Fig. 9 (0.7\").")
    parser.add_argument(
        "--paper-range", action="store_true",
        help="linear threshold axis T in [1, 20], matching the paper exactly",
    )
    parser.add_argument(
        "--variant", default="noisy", choices=("noisy", "clean"),
        help="dataset chain (default: noisy = the paper's 0.7\" case)",
    )
    args = parser.parse_args()

    # run_association_test does the full Eq. (30) optimisation for every
    # pair and prints a line each; we only want its returned (true, false)
    # L* lists, so its own output is swallowed.
    print(f"Running the all-vs-all association test ({args.variant}); ~10 s...")
    with contextlib.redirect_stdout(io.StringIO()):
        true_losses, false_losses = run_association_test(args.variant)

    print(f"  {len(true_losses)} true pairs   L* in "
          f"[{min(true_losses):.2e}, {max(true_losses):.2e}]")
    print(f"  {len(false_losses)} false pairs  L* in "
          f"[{min(false_losses):.2e}, {max(false_losses):.2e}]")

    if args.paper_range:
        # The paper's axis. L* is a chi-square-like quantity (4 d.o.f.), so
        # O(1-20) is its meaningful band -- but only if the optimiser
        # converged; unconverged pairs sit far outside it.
        thresholds = np.linspace(1.0, 20.0, 400)
        xscale = "linear"
    else:
        # Span the actual L* range, log-spaced -- our false-pair losses are
        # ~1e6+, so a linear 1-20 axis would show nothing happening.
        lo = min(min(true_losses), min(false_losses))
        hi = max(max(true_losses), max(false_losses))
        thresholds = np.logspace(np.log10(lo), np.log10(hi), 500)
        xscale = "log"

    fn, fp = sweep(true_losses, false_losses, thresholds)

    # A perfect gate exists iff the worst true L* is below the best false
    # L*; that (gap_lo, gap_hi) interval is where FN = FP = 0.
    gap_lo, gap_hi = max(true_losses), min(false_losses)
    if gap_lo < gap_hi:
        print(f"  FN = FP = 0 for T in ({gap_lo:.3e}, {gap_hi:.3e})")
    else:
        print("  no threshold gives FN = FP = 0 "
              f"(worst true L* {gap_lo:.2e} >= best false L* {gap_hi:.2e})")

    # FN and FP live on very different scales (FN 0-100 %, FP a few %), so
    # they get their own y-axes -- as in the paper's twin-axis panels.
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
    ax_fn.set_title(f"Fig. 9 analogue -- {args.variant} chain "
                    f"($\\sigma_{{\\alpha,\\delta}} = 0.7''$), boundary-value only")
    ax_fn.grid(True, alpha=0.25)
    ax_fn.legend(handles=[l1, l2], loc="center right", framealpha=0.9)

    fig.tight_layout()
    out = _FIG_DIR / (
        f"figure9_fp_fn_{args.variant}"
        f"{'_paper-range' if args.paper_range else ''}.png"
    )
    fig.savefig(out, dpi=150)
    print(f"wrote {out.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
