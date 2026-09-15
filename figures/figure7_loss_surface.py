"""Reproduce Siminski et al. Fig. 7 for this pipeline.

Fig. 7 is the boundary-value loss L(p, k) for ONE tracklet pair, drawn as
a heatmap over the range hypothesis p = (rho_1, rho_2), with one panel per
k in the pair's admissible grid. The paper's point: unlike the
initial-value method, each fixed-k slice has a single minimum, and the
overall solution is the k whose minimum is lowest.

    uv run python figures/figure7_loss_surface.py                 # noisy, default pair
    uv run python figures/figure7_loss_surface.py --pair 0.0-0.2 --n 60
    uv run python figures/figure7_loss_surface.py --variant clean
    uv run python figures/figure7_loss_surface.py --unweighted    # noisy, ~9x faster
    uv run python figures/figure7_loss_surface.py --all           # every pair
    uv run python figures/figure7_loss_surface.py --all --pairs true

--pair is "obj.trk-obj.trk" (default 1.0-1.2: object 1's 72 h pair, k in
{2.5, 3.0, 3.5} -- the closest analogue to the paper's k in [3, 4.5]
example). Markers: the true (rho_1, rho_2) from the synthetic truth, the
Eq. (31) circular-orbit seed, and the optimiser's result at that k.

--all sweeps every tracklet pair in the k-bounds file (--pairs {all,true,
false} filters by ground truth) and writes one figure each into
figures/figure7_all_<variant>/. It defaults to a coarser grid; the
weighted loss over ~75 pairs is otherwise several minutes.

Single pair  -> figures/figure7_loss_surface_<variant>_<pair>.png
--all        -> figures/figure7_all_<variant>/<pair>.png
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The pipeline stages resolve their imports from sibling directories, not a
# package, so this script that pulls from four of them has to put those on
# the path itself -- same pattern as optimisation/validate_optimiser.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("optimisation", "orbit", "hypothesis", "modelled", "discriminator"):
    sys.path.insert(0, str(_PROJECT_ROOT / _sub))

# _pair_inputs yields the full per-pair optimiser bundle (site pos/vel,
# attributables, covariances, bounds, k grid) exactly as stage 8 sees it,
# so this figure scores the same L the pipeline scores.
from validate_optimiser import _pair_inputs  # noqa: E402
from optimise_hypothesis import (  # noqa: E402
    optimize_hypothesis,
    circular_orbit_initial_guess,
)
from discriminator_loss import discriminator_loss, discriminator_loss_unweighted  # noqa: E402
from lambert import LambertNoSolution  # noqa: E402

_FIG_DIR = Path(__file__).resolve().parent


def _pair_tag(pair):
    """Filename-safe "obj.trk-obj.trk" label for a pair."""
    return (f"{pair['object_1']}.{pair['tracklet_1']}-"
            f"{pair['object_2']}.{pair['tracklet_2']}")


def _truth_ranges(data, pair):
    """Exact (rho_1_true, rho_2_true) for the pair, from the synthetic truth.

    Each tracklet has 5 measurements 30 s apart and its attributable epoch
    t0 is their mean -- i.e. the 3rd measurement lands exactly on t0. So the
    true geocentric object position at t0 is a stored row, no propagation
    needed: rho_i_true = || r_object(t0) - r_site(t0) ||. Columns of the
    generated-data matrix: 0:3 object position (noise-free), 8:11 site
    position, 14 time, 15 tracklet id, 16 object id.
    """
    oid, tid, tcol = data[:, 16].astype(int), data[:, 15].astype(int), data[:, 14]
    out = []
    # args_1[3] / args_2[3] is t0 for each tracklet (see the unpack below).
    for obj, trk, t0 in (
        (pair["object_1"], pair["tracklet_1"], pair["args_1"][3]),
        (pair["object_2"], pair["tracklet_2"], pair["args_2"][3]),
    ):
        row = data[(oid == obj) & (tid == trk) & np.isclose(tcol, t0)][0]
        out.append(float(np.linalg.norm(row[0:3] - row[8:11])))
    return out


def plot_pair(pair, data, variant, n, unweighted, out_path):
    """Compute and draw the Fig. 7 panels for one pair; return (out_path, note)."""
    # The clean chain has zero measurement covariance, so the weighted loss
    # (Eq. 18) would invert a singular matrix -- it MUST use the unweighted
    # form, same rule as optimise_hypothesis's LOSS_FUNCTIONS. On the noisy
    # chain --unweighted is an optional speed-up (skips the Eq. 19 Jacobian).
    unweighted = unweighted or variant == "clean"

    # args_i tuple layout (from _pair_inputs): (c_i, r_s_i, v_s_i, t0_i,
    # (rho_min_i, rho_max_i), attributable_i, Ca_i). The bounds are the
    # admissible region C (Eq. 22) -- the plot spans exactly that box.
    grid_k = pair["k_grid"]
    (c1, rs1, vs1, tt1, (r1lo, r1hi), att1, ca1) = pair["args_1"]
    (c2, rs2, vs2, tt2, (r2lo, r2hi), att2, ca2) = pair["args_2"]
    rho1_true, rho2_true = _truth_ranges(data, pair)

    r1 = np.linspace(r1lo, r1hi, n)   # rho_1 grid axis (plot x)
    r2 = np.linspace(r2lo, r2hi, n)   # rho_2 grid axis (plot y)

    def loss_at(p1, p2, k):
        """L(p, k) at one grid point; NaN where no Lambert transfer exists.

        discriminator_loss / _unweighted return one value per Lambert
        candidate (the vacant-focus pair for M > 0); the surface takes the
        best, matching optimise_hypothesis's min() over candidates.
        """
        try:
            if unweighted:
                vals = discriminator_loss_unweighted(
                    p1, p2, k, rs1, vs1, tt1, att1, rs2, vs2, tt2, att2)
            else:
                vals = discriminator_loss(
                    p1, p2, k, rs1, vs1, tt1, att1, ca1, rs2, vs2, tt2, att2, ca2)
            return min(vals)
        except LambertNoSolution:
            return np.nan

    # One log10-L grid per k. Z[i, j] = loss at (rho_1 = r1[j], rho_2 = r2[i]),
    # i.e. rows index rho_2 and columns rho_1 -- the layout pcolormesh(x, y, C)
    # expects for C shaped (len(y), len(x)).
    surfaces = []
    panel_min = []
    for k in grid_k:
        Z = np.array([[loss_at(a, b, k) for a in r1] for b in r2])
        surfaces.append(np.log10(Z))
        finite = Z[np.isfinite(Z)]
        panel_min.append(np.nan if finite.size == 0 else float(finite.min()))

    # Overlay points: the Eq. (31) circular-orbit seed the optimiser starts
    # from, and where L-BFGS-B actually ended. Both can fail to exist for a
    # given k (a(k) outside [a_min, a_max]); circular_orbit_initial_guess and
    # optimize_hypothesis raise ValueError in that case.
    seeds, opt = {}, {}
    for k in grid_k:
        try:
            s = circular_orbit_initial_guess(k, tt2 - tt1, c1, rs1, c2, rs2)
            seeds[k] = np.clip(s, [r1lo, r2lo], [r1hi, r2hi])
        except ValueError:
            seeds[k] = None
        try:
            ro1, ro2, _l, _res = optimize_hypothesis(
                variant, k, tt2 - tt1,
                *pair["args_1"], *pair["args_2"],
                a_min=pair["a_min"], a_max=pair["a_max"],
            )
            opt[k] = (ro1, ro2)
        except ValueError:
            opt[k] = None

    # One panel per k, side by side. Each gets its own colourbar because the
    # panel holding the minimum spans several orders of magnitude while the
    # others are near-flat plateaus -- a shared scale would wash it out.
    fig, axes = plt.subplots(1, len(grid_k), figsize=(4.6 * len(grid_k), 4.4),
                             squeeze=False)
    for ax, k, Z in zip(axes[0], grid_k, surfaces):
        # Ranges drawn in 10^4 km to match the paper's axes.
        im = ax.pcolormesh(r1 / 1e4, r2 / 1e4, Z, shading="auto", cmap="Greys_r")
        fig.colorbar(im, ax=ax, label=r"$\log_{10} L(\mathbf{p})$")
        ax.plot(rho1_true / 1e4, rho2_true / 1e4, "o", ms=9, mfc="none",
                mec="#d1495b", mew=1.8, label="truth")
        if seeds[k] is not None:
            ax.plot(seeds[k][0] / 1e4, seeds[k][1] / 1e4, "P", ms=8,
                    color="#3a7ca5", label="Eq. 31 seed")
        if opt[k] is not None:
            ax.plot(opt[k][0] / 1e4, opt[k][1] / 1e4, "x", ms=9, mew=2,
                    color="#2a9d3a", label="optimiser L*")
        ax.set_title(f"$k \\in [{k}, {k + 0.5}]$")
        ax.set_xlabel(r"$\rho_1$  [$10^4$ km]")
        ax.set_ylabel(r"$\rho_2$  [$10^4$ km]")
    axes[0][0].legend(loc="upper left", fontsize=8, framealpha=0.9)

    kind = "TRUE" if pair["is_true"] else "false"
    fig.suptitle(f"Fig. 7 analogue -- {_pair_tag(pair)} [{kind}], {variant} chain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    best = int(np.nanargmin(panel_min))
    note = (f"min log10 L = {panel_min[best]:.2f} at k={grid_k[best]}"
            if np.isfinite(panel_min[best]) else "no feasible point")
    return out_path, note


def main():
    parser = argparse.ArgumentParser(description="Reproduce Siminski Fig. 7.")
    parser.add_argument("--pair", default="1.0-1.2", help='"obj.trk-obj.trk"')
    parser.add_argument("--variant", default="noisy", choices=("noisy", "clean"))
    parser.add_argument("--n", type=int, default=None,
                        help="grid points per axis (default 55, or 40 with --all)")
    parser.add_argument("--unweighted", action="store_true",
                        help="noisy chain: use the unweighted loss for speed "
                             "(no Eq. 19 Jacobian, ~9x faster). The clean chain "
                             "always uses it -- its covariance is singular.")
    parser.add_argument("--all", action="store_true",
                        help="one figure per tracklet pair, into figures/figure7_all_<variant>/")
    parser.add_argument("--pairs", default="all", choices=("all", "true", "false"),
                        help="with --all, which pairs to plot (by ground truth)")
    args = parser.parse_args()

    n = args.n if args.n is not None else (40 if args.all else 55)

    from range_bounds import GENERATED_DATA_FILES  # noqa: E402
    with np.load(GENERATED_DATA_FILES[args.variant]) as archive:
        data = archive["data"]

    if not args.all:
        # "obj.trk-obj.trk", e.g. "1.0-1.2" -> object 1 tracklet 0 vs tracklet 2.
        (o1, t1), (o2, t2) = (map(int, p.split(".")) for p in args.pair.split("-"))
        pair = next(
            (p for p in _pair_inputs(args.variant)
             if (p["object_1"], p["tracklet_1"], p["object_2"], p["tracklet_2"])
             == (o1, t1, o2, t2)),
            None,
        )
        if pair is None:
            raise SystemExit(f"pair {args.pair} not in the {args.variant} k-bounds file")
        out = _FIG_DIR / f"figure7_loss_surface_{args.variant}_{args.pair}.png"
        print(f"pair {args.pair} ({args.variant}); k grid {pair['k_grid']}; "
              f"{n}x{n} grid...")
        _, note = plot_pair(pair, data, args.variant, n, args.unweighted, out)
        print(f"  {note}\nwrote {out.relative_to(_PROJECT_ROOT)}")
        return

    pairs = [
        p for p in _pair_inputs(args.variant)
        if args.pairs == "all"
        or (args.pairs == "true") == bool(p["is_true"])
    ]
    out_dir = _FIG_DIR / f"figure7_all_{args.variant}"
    out_dir.mkdir(exist_ok=True)
    loss_name = "unweighted" if (args.unweighted or args.variant == "clean") else "weighted"
    print(f"{len(pairs)} {args.pairs} pairs, {n}x{n} grid, {loss_name} loss "
          f"-> {out_dir.relative_to(_PROJECT_ROOT)}/")

    started = time.monotonic()
    for i, pair in enumerate(pairs, 1):
        tag = _pair_tag(pair)
        out, note = plot_pair(
            pair, data, args.variant, n, args.unweighted, out_dir / f"{tag}.png",
        )
        kind = "TRUE " if pair["is_true"] else "false"
        print(f"  [{i:2d}/{len(pairs)}] {kind} {tag:12s}  {note}")
    print(f"done in {time.monotonic() - started:.0f}s")


if __name__ == "__main__":
    main()
