"""Stages [5] and [6]: candidate orbits and the angular rates they predict.

Solves Lambert for every tracklet pair and every k in its admissible grid, at
the midpoint of each tracklet's range bounds, and saves one row per candidate
orbit to model_<variant>.npz. The midpoint is an illustrative hypothesis, not
the Eq. (31) seed or an optimised one: this shows the Model phase on its own,
and is not an association test.

    uv run python experiment/run/run_model.py --parent <hypothesis hash>
    uv run python experiment/run/run_model.py -n 4
"""

import numpy as np

import lab.paths
import run_hypothesis
from lab.runner import execute, stage_parser


def model_rows(pairs):
    """One row per feasible (pair, k, candidate)."""
    from lab.model.build_orbit import build_orbit
    from lab.model.lambert import LambertNoSolution
    from lab.model.model_rates import modeled_angular_rates

    rows = []
    infeasible = 0
    for pair in pairs:
        (_c1, rs1, vs1, t1, (r1lo, r1hi), *_rest1) = pair["args_1"]
        (_c2, rs2, vs2, t2, (r2lo, r2hi), *_rest2) = pair["args_2"]
        rho_1, rho_2 = 0.5 * (r1lo + r1hi), 0.5 * (r2lo + r2hi)

        for k in pair["k_grid"]:
            try:
                r_1, r_2, _dt, candidates = build_orbit(
                    rs1, pair["u_1"], t1, rho_1, rs2, pair["u_2"], t2, rho_2, k,
                )
            except LambertNoSolution:
                infeasible += 1
                continue

            for index, solution in enumerate(candidates):
                rates = modeled_angular_rates(
                    r_1, solution.v1, rs1, vs1, r_2, solution.v2, rs2, vs2,
                )
                rows.append((
                    pair["object_1"], pair["tracklet_1"],
                    pair["object_2"], pair["tracklet_2"],
                    pair["is_true"], k, index, rho_1, rho_2,
                    *solution.v1, *solution.v2, *rates,
                ))
    return np.array(rows, dtype=float), infeasible


def run(parent, force=False):
    def body(tee, output_dir, config):
        # The stage modules read their file locations from lab.paths when first
        # imported, so point it at this run before importing them.
        lab.paths.DATA_DIR = output_dir
        from lab.evaluation.validate_optimiser import _pair_inputs
        from lab.hypothesis.range_bounds import (
            PRECISION_OUTPUT_FILES,
            VARIANTS,
        )

        config["candidates"] = {}
        for variant in VARIANTS:
            with np.load(PRECISION_OUTPUT_FILES[variant]) as archive:
                ids = list(zip(archive["object_id"].tolist(), archive["tracklet_id"].tolist()))
                u_i = archive["u_i"]

            pairs = []
            for pair in _pair_inputs(variant):
                pair["u_1"] = u_i[ids.index((pair["object_1"], pair["tracklet_1"]))]
                pair["u_2"] = u_i[ids.index((pair["object_2"], pair["tracklet_2"]))]
                pairs.append(pair)

            rows, infeasible = model_rows(pairs)
            np.savez(
                output_dir / f"model_{variant}.npz",
                object_id_1=rows[:, 0].astype(int), tracklet_id_1=rows[:, 1].astype(int),
                object_id_2=rows[:, 2].astype(int), tracklet_id_2=rows[:, 3].astype(int),
                is_true=rows[:, 4].astype(bool), k=rows[:, 5],
                candidate=rows[:, 6].astype(int),
                rho_1_km=rows[:, 7], rho_2_km=rows[:, 8],
                v1_km_s=rows[:, 9:12], v2_km_s=rows[:, 12:15],
                z_hat_dot=rows[:, 15:19],
            )
            config["candidates"][variant] = {
                "candidates": len(rows), "infeasible_k": infeasible,
            }
            tee.write(f"{variant}: {len(rows)} candidate orbits, "
                      f"{infeasible} infeasible k\n")
        return True

    return execute("model", {}, (parent, "hypothesis"), body, force)


def build(objects):
    return run(run_hypothesis.build(objects))


if __name__ == "__main__":
    args = stage_parser(__doc__, "hypothesis").parse_args()
    run(args.parent or run_hypothesis.build(args.objects), args.force)
