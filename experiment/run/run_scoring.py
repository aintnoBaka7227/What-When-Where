"""Stage [7]: the loss L(p, k) over a grid of range hypotheses, per tracklet pair.

Saves one surface_<variant>_<pair>.npz per pair, the data behind Fig. 7.

    uv run python experiment/run/run_scoring.py --parent <hypothesis hash> --pair 0.0-0.2
    uv run python experiment/run/run_scoring.py --parent <hypothesis hash> --all --pairs true
    uv run python experiment/run/run_scoring.py -n 4 --pair 0.0-0.2
"""

import numpy as np

import lab.paths
import run_hypothesis
from lab.data.dataset import VARIANTS, load_data
from lab.runner import execute, stage_parser, write_metrics


def _pair_tag(pair):
    """Filename-safe "obj.trk-obj.trk" label for a pair."""
    return (f"{pair['object_1']}.{pair['tracklet_1']}-"
            f"{pair['object_2']}.{pair['tracklet_2']}")


def _truth_ranges(data, pair):
    """Exact (rho_1_true, rho_2_true) for the pair, from the synthetic truth.

    Each tracklet has 5 measurements 20 s apart and its attributable epoch
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


def compute_loss_surface(pair, data, variant, n, unweighted):
    """The Fig. 7 surfaces for one pair, as arrays for experiment/prod/prod_scoring.py."""
    from lab.search.optimise_hypothesis import (
        optimize_hypothesis,
        circular_orbit_initial_guess,
    )
    from lab.scoring.discriminator_loss import discriminator_loss, discriminator_loss_unweighted
    from lab.model.lambert import LambertNoSolution

    # The clean chain has zero measurement covariance, so the weighted loss
    # (Eq. 18) would invert a singular matrix -- it MUST use the unweighted
    # form, same rule as optimise_hypothesis's LOSS_FUNCTIONS. On the noisy
    # chain --unweighted is an optional speed-up (skips the Eq. 19 Jacobian).
    unweighted = unweighted or variant == "clean"

    # args_i tuple layout (from _pair_inputs): (c_i, r_s_i, v_s_i, t0_i,
    # (rho_min_i, rho_max_i), attributable_i, Ca_i). The bounds are the
    # admissible region C (Eq. 22) -- the surface spans exactly that box.
    grid_k = pair["k_grid"]
    (c1, rs1, vs1, tt1, (r1lo, r1hi), att1, ca1) = pair["args_1"]
    (c2, rs2, vs2, tt2, (r2lo, r2hi), att2, ca2) = pair["args_2"]

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

    # One L grid per k. Z[i, j] = loss at (rho_1 = r1[j], rho_2 = r2[i]),
    # i.e. rows index rho_2 and columns rho_1 -- the layout pcolormesh(x, y, C)
    # expects for C shaped (len(y), len(x)).
    surfaces = np.array([
        [[loss_at(a, b, k) for a in r1] for b in r2] for k in grid_k
    ])

    # Overlay points: the Eq. (31) circular-orbit seed the optimiser starts
    # from, and where L-BFGS-B actually ended. Both can fail to exist for a
    # given k (a(k) outside [a_min, a_max]); circular_orbit_initial_guess and
    # optimize_hypothesis raise ValueError in that case, stored here as NaN.
    seeds = np.full((len(grid_k), 2), np.nan)
    optima = np.full((len(grid_k), 2), np.nan)
    for index, k in enumerate(grid_k):
        try:
            s = circular_orbit_initial_guess(k, tt2 - tt1, c1, rs1, c2, rs2)
            seeds[index] = np.clip(s, [r1lo, r2lo], [r1hi, r2hi])
        except ValueError:
            pass
        try:
            ro1, ro2, _l, _res = optimize_hypothesis(
                variant, k, tt2 - tt1,
                *pair["args_1"], *pair["args_2"],
                a_min=pair["a_min"], a_max=pair["a_max"],
            )
            optima[index] = (ro1, ro2)
        except ValueError:
            pass

    return {
        "variant": variant,
        "tag": _pair_tag(pair),
        "is_true": pair["is_true"],
        "k_grid": np.array(grid_k, dtype=float),
        "rho_1": r1,
        "rho_2": r2,
        "loss": surfaces,
        "truth": np.array(_truth_ranges(data, pair)),
        "seeds": seeds,
        "optima": optima,
    }


def run(parent, variant="noisy_0.7as", n=55, unweighted=False, pair=None, pairs=None, force=False):
    params = {
        "variant": variant, "n": n, "unweighted": unweighted,
        "pair": pair, "pairs": pairs,
    }

    def body(tee, output_dir, config):
        # The stage modules read their file locations from lab.paths when first
        # imported, so point it at this run before importing them.
        lab.paths.DATA_DIR = output_dir
        from lab.evaluation.metrics import scoring_metrics
        from lab.evaluation.validate_optimiser import _pair_inputs
        from lab.scoring.discriminator_loss import evaluate_loss

        data = load_data(variant)

        if pair is None:
            selected = [
                p for p in _pair_inputs(variant)
                if pairs == "all" or (pairs == "true") == bool(p["is_true"])
            ]
        else:
            selected = [p for p in _pair_inputs(variant) if _pair_tag(p) == pair]
            if not selected:
                tee.write(f"pair {pair} not in the {variant} k-bounds file\n")
                return False

        for index, current in enumerate(selected, 1):
            surface = compute_loss_surface(current, data, variant, n, unweighted)
            np.savez(output_dir / f"surface_{variant}_{surface['tag']}.npz", **surface)

            finite = surface["loss"][np.isfinite(surface["loss"])]
            note = (f"min log10 L = {np.log10(finite.min()):.2f}"
                    if finite.size else "no feasible point")
            kind = "TRUE " if current["is_true"] else "false"
            tee.write(f"  [{index:2d}/{len(selected)}] {kind} {surface['tag']:12s}  {note}\n")

        write_metrics(output_dir, scoring_metrics(output_dir, _pair_inputs, evaluate_loss))
        return True

    return execute("scoring", params, (parent, "hypothesis"), body, force)


if __name__ == "__main__":
    parser = stage_parser(__doc__, "hypothesis")
    parser.add_argument("--pair", default="0.0-0.2", help='"obj.trk-obj.trk"')
    parser.add_argument("--variant", default="noisy_0.7as", choices=VARIANTS)
    parser.add_argument("--n", type=int, default=None,
                        help="grid points per axis (default 55, or 40 with --all)")
    parser.add_argument("--unweighted", action="store_true",
                        help="noisy chain: use the unweighted loss for speed "
                             "(no Eq. 19 Jacobian, ~9x faster). The clean chain "
                             "always uses it -- its covariance is singular.")
    parser.add_argument("--all", action="store_true", help="every tracklet pair")
    parser.add_argument("--pairs", default="all", choices=("all", "true", "false"),
                        help="with --all, which pairs to compute (by ground truth)")
    args = parser.parse_args()

    run(
        args.parent or run_hypothesis.build(args.objects),
        variant=args.variant,
        n=args.n if args.n is not None else (40 if args.all else 55),
        unweighted=args.unweighted,
        pair=None if args.all else args.pair,
        pairs=args.pairs if args.all else None,
        force=args.force,
    )
