"""Step 1 of validating stage 8: did the L(p, k) minimiser actually converge?

`optimise_hypothesis.py` reports one number per pair, L*. That number is
only trustworthy if L-BFGS-B genuinely walked downhill to it. Near a
half-revolution transfer the loss valley is a narrow trench, the Eq. (31)
circular seed lands on its wall, and the default L-BFGS-B stopping test can
declare "converged" after two steps -- thousands of times above the true
minimum (see AGENTS.md, "One true pair sometimes converges on the wrong k").

For every tracklet pair and every k in its admissible grid, this re-runs
the same optimisation `run_association_test` does, but with a `history`
list attached, and reports:

    L_seed        loss at the Eq. (31) circular-orbit initial guess
    L_final       loss L-BFGS-B stopped at
    drop          L_seed / L_final -- how much it actually removed
    nit / nfev    iterations and function evaluations it took
    message       L-BFGS-B's own termination reason
    best-so-far   log10 of the running-minimum loss, subsampled -- the curve

Per (pair, k), a marker is raised when:

    seed infeasible   the circular seed itself has no Lambert solution
                      (loss pinned at the infeasibility penalty)
    no step taken     L-BFGS-B ran >= 20 evaluations -- so its gradient said
                      to move -- but completed <= 1 iteration: every line
                      search failed (bad scaling / a cliff).

Then, per pair, the k with the lowest L_final is the one stage 8 keeps.
That winning run is flagged "weak descent" when it is a TRUE pair and the
search only divided the seed loss by < 20x: every genuinely converged TRUE
pair in this dataset divides it by > 1000x, so < 20x means L-BFGS-B never
reached the valley. This is what catches the Object 0 failures.

None of these prove the returned L* is wrong: a genuinely flat region
(a false pair, or a wrong k) also produces a short, near-horizontal curve,
and a TRUE pair whose seed already sat in the valley descends little yet
ends correct. Deciding whether a short curve is a real failure needs the
truth-range comparison in step 3. This script only shows whether, and how
far, the optimiser moved.

Every run is captured to runs/, tied to a timestamp (or the --stamp
`main.py --validate` passes so it matches that pipeline run's log):

    runs/<stamp>-validate.log               the terminal output, verbatim
    runs/optimiser_history_<variant>_<stamp>.csv   per-evaluation trace
        (columns pair, object/tracklet ids, is_true, k, eval, rho_1,
         rho_2, loss) -- for plotting eval-number against log10(L) elsewhere

    uv run python src/lab/evaluation/validate_optimiser.py            # own timestamp
    uv run python src/lab/evaluation/validate_optimiser.py --no-log   # terminal only

This does not fix anything; it is a diagnostic you run when a true pair's
L* looks wrong.
"""

import argparse
import contextlib
import csv
import sys
from datetime import datetime

import numpy as np

from lab.hypothesis.range_bounds import (
    VARIANTS,
    GENERATED_DATA_FILES,
    ATTRIBUTABLES_FILES,
    population_for,
    v_s_i_at_epoch,
)
from lab.hypothesis.k_bounds import k_grid
from lab.model.build_orbit import RANGE_BOUNDS_FILES, K_BOUNDS_FILES, _lookup_tracklet
from lab.search.optimise_hypothesis import (
    optimize_hypothesis,
    _INFEASIBLE_PENALTY,
)
from lab.paths import PROJECT_ROOT, RUNS_DIR

# "No step taken": L-BFGS-B evaluated the objective at least this many times
# -- its finite-difference gradient told it to move -- yet completed at most
# this many iterations. Every trial step was rejected.
_NOSTEP_MIN_NFEV = 20
_NOSTEP_MAX_NIT = 1


class _Tee:
    """A stdout stand-in that also copies everything to a log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, text):
        self._stream.write(text)
        self._log.write(text)

    def flush(self):
        self._stream.flush()
        self._log.flush()


@contextlib.contextmanager
def _capture_to(log_file):
    """Redirect print() to both the terminal and log_file for the block."""
    if log_file is None:
        yield
        return
    original = sys.stdout
    sys.stdout = _Tee(original, log_file)
    try:
        yield
    finally:
        sys.stdout = original


def _pair_inputs(variant):
    """Yield the full optimiser input bundle for every tracklet pair.

    Mirrors the loading in optimise_hypothesis.run_association_test so this
    diagnostic scores exactly what the pipeline scores.
    """
    a_min, a_max, _ = population_for(variant)

    with np.load(GENERATED_DATA_FILES[variant]) as archive:
        data = archive["data"]

    with np.load(RANGE_BOUNDS_FILES[variant]) as archive:
        rb_object_ids = archive["object_id"]
        rb_tracklet_ids = archive["tracklet_id"]
        t0_values = archive["t0"]
        r_s_i = archive["r_s_i_km"]
        c_i = archive["c_i_km"]
        rho_min_i = archive["rho_min_i_km"]
        rho_max_i = archive["rho_max_i_km"]

    with np.load(K_BOUNDS_FILES[variant]) as archive:
        pair_object_id_1 = archive["object_id_1"]
        pair_tracklet_id_1 = archive["tracklet_id_1"]
        pair_object_id_2 = archive["object_id_2"]
        pair_tracklet_id_2 = archive["tracklet_id_2"]
        pair_k_min = archive["k_min"]
        pair_k_max = archive["k_max"]

    with np.load(ATTRIBUTABLES_FILES[variant]) as archive:
        attr_object_ids = archive["object_id"]
        attr_tracklet_ids = archive["tracklet_id"]
        attributables = archive["attributable"]
        covariances = archive["Ca"]

    for idx in range(len(pair_object_id_1)):
        i = _lookup_tracklet(
            rb_object_ids, rb_tracklet_ids,
            pair_object_id_1[idx], pair_tracklet_id_1[idx],
        )
        j = _lookup_tracklet(
            rb_object_ids, rb_tracklet_ids,
            pair_object_id_2[idx], pair_tracklet_id_2[idx],
        )
        ai = _lookup_tracklet(
            attr_object_ids, attr_tracklet_ids,
            rb_object_ids[i], rb_tracklet_ids[i],
        )
        aj = _lookup_tracklet(
            attr_object_ids, attr_tracklet_ids,
            rb_object_ids[j], rb_tracklet_ids[j],
        )

        yield {
            "idx": idx,
            "object_1": int(rb_object_ids[i]),
            "tracklet_1": int(rb_tracklet_ids[i]),
            "object_2": int(rb_object_ids[j]),
            "tracklet_2": int(rb_tracklet_ids[j]),
            "is_true": int(rb_object_ids[i]) == int(rb_object_ids[j]),
            "delta_t": abs(t0_values[j] - t0_values[i]),
            "k_grid": list(k_grid(pair_k_min[idx], pair_k_max[idx])),
            "a_min": a_min,
            "a_max": a_max,
            "args_1": (
                c_i[i], r_s_i[i],
                v_s_i_at_epoch(data, rb_object_ids[i], rb_tracklet_ids[i], t0_values[i]),
                t0_values[i], (rho_min_i[i], rho_max_i[i]),
                attributables[ai], covariances[ai],
            ),
            "args_2": (
                c_i[j], r_s_i[j],
                v_s_i_at_epoch(data, rb_object_ids[j], rb_tracklet_ids[j], t0_values[j]),
                t0_values[j], (rho_min_i[j], rho_max_i[j]),
                attributables[aj], covariances[aj],
            ),
        }


def _running_min(losses):
    """Best-so-far loss after each evaluation -- the curve to plot."""
    best = np.inf
    out = []
    for value in losses:
        best = min(best, value)
        out.append(best)
    return out


def _format_trace(running_min, points=16):
    """log10 of the best-so-far curve, evenly subsampled to `points` values."""
    n = len(running_min)
    if n <= points:
        picks = range(n)
    else:
        picks = sorted(set(int(round(i * (n - 1) / (points - 1))) for i in range(points)))
    return " ".join(
        f"{np.log10(running_min[i]):5.1f}" if running_min[i] > 0 else "  -inf"
        for i in picks
    )


def validate_variant(variant, stamp, write_csv=True):
    """Trace and report every optimisation for one dataset chain."""
    csv_path = (
        RUNS_DIR / f"optimiser_history_{variant}_{stamp}.csv" if write_csv else None
    )
    if write_csv:
        RUNS_DIR.mkdir(exist_ok=True)

    n_runs = 0
    winners = []  # (label, is_true, best_k, best_loss, best_marker)

    with (csv_path.open("w", newline="") if write_csv else contextlib.nullcontext()) as handle:
        writer = csv.writer(handle) if write_csv else None
        if writer is not None:
            writer.writerow([
                "pair", "object_1", "tracklet_1", "object_2", "tracklet_2",
                "is_true", "k", "eval", "rho_1", "rho_2", "loss",
            ])

        for pair in _pair_inputs(variant):
            label = (
                f"Object {pair['object_1']} tracklet {pair['tracklet_1']} <-> "
                f"Object {pair['object_2']} tracklet {pair['tracklet_2']} "
                f"[{'TRUE' if pair['is_true'] else 'false'}]"
            )
            print(label)
            best = None

            for k in pair["k_grid"]:
                history = []
                try:
                    rho_1_opt, rho_2_opt, loss_final, result = optimize_hypothesis(
                        variant, k, pair["delta_t"],
                        *pair["args_1"],
                        *pair["args_2"],
                        a_min=pair["a_min"], a_max=pair["a_max"],
                        history=history,
                    )
                except ValueError as exc:
                    print(f"  k={k}: skipped ({exc})")
                    continue

                n_runs += 1
                losses = [entry[2] for entry in history]
                running_min = _running_min(losses)
                l_seed = losses[0]
                drop = l_seed / loss_final if loss_final > 0 else np.inf

                if l_seed >= _INFEASIBLE_PENALTY:
                    marker = "seed infeasible"
                elif result.nfev >= _NOSTEP_MIN_NFEV and result.nit <= _NOSTEP_MAX_NIT:
                    marker = "no step taken"
                else:
                    marker = ""
                tag = f"  <-- {marker}" if marker else ""

                if best is None or loss_final < best[1]:
                    best = (k, loss_final, marker, l_seed, drop, result.nit)

                print(
                    f"  k={k}: L_seed={l_seed:.3e} -> L_final={loss_final:.3e}  "
                    f"drop={drop:8.1f}x  nit={result.nit} nfev={result.nfev}{tag}"
                )
                print(
                    f"      rho*=({rho_1_opt:.1f}, {rho_2_opt:.1f}) km  "
                    f"success={result.success}"
                )
                print(f"      log10(best-so-far): {_format_trace(running_min)}")
                print(f"      stop: {result.message}")

                if writer is not None:
                    for eval_index, (rho_1, rho_2, loss) in enumerate(history):
                        writer.writerow([
                            pair["idx"], pair["object_1"], pair["tracklet_1"],
                            pair["object_2"], pair["tracklet_2"], pair["is_true"],
                            k, eval_index, f"{rho_1:.6f}", f"{rho_2:.6f}", f"{loss:.10e}",
                        ])

            if best is not None:
                winners.append((label, pair["is_true"], *best))

    def _weak(w):
        # Winning run barely descended. Every genuinely converged TRUE pair
        # here divides its seed loss by >1000x; a factor under 20x means the
        # search never reached the valley. (Only meaningful for TRUE pairs --
        # a false pair has no valley to reach.)
        _label, is_true, _k, _loss, marker, _seed, drop, _nit = w
        return is_true and not marker and drop < 20.0

    print()
    print("Winning k per pair (what stage 8 reports):")
    for w in winners:
        label, _is_true, k, loss, marker, l_seed, drop, nit = w
        note = f"  <-- {marker}" if marker else ("  <-- weak descent" if _weak(w) else "")
        print(f"  {label}  k*={k}  L*={loss:.3e}  "
              f"(seed {l_seed:.1e}, {drop:.0f}x in {nit} it){note}")

    suspect = [w for w in winners if (w[1] and w[4]) or _weak(w)]
    print()
    print(f"{len(suspect)} TRUE pair(s) whose winning run looks unconverged:")
    for w in suspect:
        label, _is_true, k, loss, marker, _l_seed, drop, nit = w
        why = marker or f"weak descent ({drop:.0f}x in {nit} it)"
        print(f"  {label}  k*={k}  L*={loss:.3e}  ({why})")
    print()
    print("These are heuristics on the trajectory, not proof L* is wrong: a")
    print("false pair or a wrong k produces the same short, flat curve. Confirm")
    print("with step 3 -- compare L* against L at the known true ranges.")
    if csv_path is not None:
        print(f"per-evaluation trace: {csv_path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description="Trace every stage-8 optimisation and report convergence.",
    )
    parser.add_argument(
        "--stamp", default=None,
        help="timestamp tag for the output files; defaults to now, and "
             "main.py --validate passes its run's stamp so they match that log",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="print to the terminal only; do not write runs/<stamp>-validate.log",
    )
    args = parser.parse_args()

    stamp = args.stamp or datetime.now().strftime("%Y-%m-%d-%H%M%S")

    log_file = None
    if not args.no_log:
        RUNS_DIR.mkdir(exist_ok=True)
        log_file = (RUNS_DIR / f"{stamp}-validate.log").open("w")

    with _capture_to(log_file):
        if log_file is not None:
            rule = "=" * 68
            print(f"{rule}\n optimiser validation -- {stamp}\n{rule}")

        for variant in VARIANTS:
            if not (
                GENERATED_DATA_FILES[variant].exists()
                and ATTRIBUTABLES_FILES[variant].exists()
                and RANGE_BOUNDS_FILES[variant].exists()
                and K_BOUNDS_FILES[variant].exists()
            ):
                print(f"Skipping {variant}: inputs missing")
                continue

            print(f"--- {variant} ---")
            validate_variant(variant, stamp, write_csv=not args.no_log)
            print()

        if log_file is not None:
            print(f"log: runs/{stamp}-validate.log")

    if log_file is not None:
        log_file.close()


if __name__ == "__main__":
    main()
