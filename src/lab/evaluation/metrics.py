"""Metrics for judging each pipeline stage against the simulated truth.

Each function reads a run's output folder and returns a plain dict, which
experiment/run/ saves as metrics.json in that run's folder:

    hypothesis_metrics   is the truth inside the admissible region? (Eqs. 22, 29)
    scoring_metrics      does the loss score the truth as a match? (Eq. 18)
    search_metrics       did the optimiser reach the right answer? (Eq. 30)
    decision_metrics     how well does a threshold separate pairs? (Fig. 9)

Truth comes from columns 0-5 of the data matrix, which only metrics may read.
Those columns are the same in every chain, so they are read from clean.
"""

import numpy as np
from astropy.constants import GM_earth

from lab.data.dataset import VARIANTS

# km^3/s^2, matches the rest of the project.
MU = GM_earth.value / 1_000_000_000.0

# The optimiser's penalty for a hypothesis with no Lambert solution, matching
# _INFEASIBLE_PENALTY in src/lab/search/optimise_hypothesis.py.
INFEASIBLE_PENALTY = 1.0e12

# Fig. 9's threshold axis runs from 1 to 20. L* is only on that scale in the
# noisy chains; the clean loss has different units.
PAPER_THRESHOLDS = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0)


def true_k(r1, v1, r2, delta_t, mu=MU):
    """The k of Eq. (29) that the true orbit has between two positions.

    The integer part is the number of completed revolutions, and the fraction
    is 0.5 for a transfer angle above 180 degrees (see k_to_revolutions).
    """
    h = np.cross(r1, v1)
    a = 1.0 / (2.0 / np.linalg.norm(r1) - np.dot(v1, v1) / mu)
    period = 2.0 * np.pi * np.sqrt(a**3 / mu)

    swept = np.arctan2(
        np.dot(h / np.linalg.norm(h), np.cross(r1, r2)),
        np.dot(r1, r2),
    ) % (2.0 * np.pi)
    revolutions = round(delta_t / period - swept / (2.0 * np.pi))

    return revolutions + (0.5 if swept > np.pi else 0.0)


class Truth:
    """True states at each tracklet's epoch, from a run's output folder."""

    def __init__(self, output_dir):
        self.rows = np.loadtxt(
            output_dir / "generated_data_gcrs_clean.txt", delimiter=",", ndmin=2,
        )
        with np.load(output_dir / "range_bounds_clean.npz") as archive:
            self.epochs = {
                (int(obj), int(trk)): float(t0)
                for obj, trk, t0 in zip(
                    archive["object_id"], archive["tracklet_id"], archive["t0"],
                )
            }

    def row(self, object_id, tracklet_id):
        """The row at the tracklet's epoch, which is its middle observation."""
        t0 = self.epochs[(object_id, tracklet_id)]
        match = (
            (self.rows[:, 16] == object_id)
            & (self.rows[:, 15] == tracklet_id)
            & np.isclose(self.rows[:, 14], t0)
        )
        return self.rows[match][0]

    def range_km(self, object_id, tracklet_id):
        row = self.row(object_id, tracklet_id)
        return float(np.linalg.norm(row[:3] - row[8:11]))

    def k(self, object_id, tracklet_1, tracklet_2):
        row_1 = self.row(object_id, tracklet_1)
        row_2 = self.row(object_id, tracklet_2)
        if row_2[14] < row_1[14]:
            row_1, row_2 = row_2, row_1
        return true_k(row_1[:3], row_1[3:6], row_2[:3], row_2[14] - row_1[14])


def share(count, total):
    return float(count / total) if total else None


def hypothesis_metrics(output_dir):
    """Metrics 1 and 2: the truth must be inside the region Search can reach."""
    truth = Truth(output_dir)
    metrics = {}

    for variant in VARIANTS:
        with np.load(output_dir / f"range_bounds_{variant}.npz") as archive:
            margins = [
                min(rho - rho_min, rho_max - rho)
                for rho, rho_min, rho_max in (
                    (truth.range_km(int(obj), int(trk)), lo, hi)
                    for obj, trk, lo, hi in zip(
                        archive["object_id"], archive["tracklet_id"],
                        archive["rho_min_i_km"], archive["rho_max_i_km"],
                    )
                )
            ]
        margins = np.array(margins)

        with np.load(output_dir / f"k_bounds_{variant}.npz") as archive:
            pairs = [
                (int(o1), int(t1), int(t2), k_min, k_max)
                for o1, t1, o2, t2, k_min, k_max in zip(
                    archive["object_id_1"], archive["tracklet_id_1"],
                    archive["object_id_2"], archive["tracklet_id_2"],
                    archive["k_min"], archive["k_max"],
                )
                if o1 == o2
            ]
        k_inside = sum(
            k_min <= truth.k(obj, t1, t2) <= k_max
            for obj, t1, t2, k_min, k_max in pairs
        )

        metrics[variant] = {
            "true_range_inside_bounds": {
                "tracklets": len(margins),
                "inside": int(np.sum(margins >= 0.0)),
                "share": share(np.sum(margins >= 0.0), len(margins)),
                "smallest_margin_km": float(margins.min()),
            },
            "true_k_inside_grid": {
                "true_pairs": len(pairs),
                "inside": int(k_inside),
                "share": share(k_inside, len(pairs)),
            },
        }

    return metrics


def scoring_metrics(output_dir, pair_inputs, evaluate_loss):
    """Metric 3: the loss at the true ranges and k, for every true pair.

    pair_inputs and evaluate_loss are passed in, because they read file
    locations when imported. Eq. (18) is chi-squared-like with 4 degrees of
    freedom, so on a noisy chain the mean should be near 4.
    """
    from lab.model.lambert import LambertNoSolution

    truth = Truth(output_dir)
    metrics = {}

    for variant in VARIANTS:
        losses = []
        no_solution = 0

        for pair in pair_inputs(variant):
            if not pair["is_true"]:
                continue

            obj = pair["object_1"]
            (_c1, r_s1, v_s1, t1, _b1, attributable_1, ca_1) = pair["args_1"]
            (_c2, r_s2, v_s2, t2, _b2, attributable_2, ca_2) = pair["args_2"]
            try:
                losses.append(min(evaluate_loss(
                    variant,
                    truth.range_km(obj, pair["tracklet_1"]),
                    truth.range_km(obj, pair["tracklet_2"]),
                    truth.k(obj, pair["tracklet_1"], pair["tracklet_2"]),
                    r_s1, v_s1, t1, attributable_1, ca_1,
                    r_s2, v_s2, t2, attributable_2, ca_2,
                )))
            except LambertNoSolution:
                no_solution += 1

        losses = np.array(losses)
        metrics[variant] = {
            "loss_at_truth": {
                "true_pairs": len(losses) + no_solution,
                "no_lambert_solution": no_solution,
                "mean": float(losses.mean()),
                "median": float(np.median(losses)),
                "max": float(losses.max()),
            },
        }

    return metrics


def search_metrics(output_dir):
    """Metrics 4, 5 and 6: whether each search worked, not just what it returned."""
    truth = Truth(output_dir)
    metrics = {}

    for variant in VARIANTS:
        with np.load(output_dir / f"association_{variant}.npz") as archive:
            is_true = archive["is_true"]
            pairs = [
                (int(o1), int(t1), int(t2), k)
                for o1, t1, t2, k, same in zip(
                    archive["object_id_1"], archive["tracklet_id_1"],
                    archive["tracklet_id_2"], archive["k"], is_true,
                )
                if same
            ]
            total_pairs = len(is_true)
        correct = sum(k == truth.k(obj, t1, t2) for obj, t1, t2, k in pairs)

        with np.load(output_dir / f"search_trace_{variant}.npz") as trace:
            seed_loss = trace["seed_loss"]
            iterations = trace["iterations"]
            evaluations = trace["evaluations"]
            pair_index = trace["pair"]
            true_search = trace["is_true"]

        penalty_start = seed_loss >= INFEASIBLE_PENALTY
        per_pair = np.bincount(pair_index, weights=evaluations, minlength=total_pairs)

        metrics[variant] = {
            "correct_k": {
                "true_pairs": len(pairs),
                "correct": int(correct),
                "share": share(correct, len(pairs)),
            },
            "penalty_starts": {
                "searches": len(seed_loss),
                "all": int(penalty_start.sum()),
                "true_pairs": int(np.sum(penalty_start & true_search)),
                "false_pairs": int(np.sum(penalty_start & ~true_search)),
                "share": share(penalty_start.sum(), len(seed_loss)),
            },
            "iterations": {
                "median": float(np.median(iterations)),
                "p95": float(np.percentile(iterations, 95)),
                "max": int(iterations.max()),
                "zero_iteration_searches": int(np.sum(iterations == 0)),
            },
            "loss_evaluations_per_pair": {
                "median": float(np.median(per_pair)),
                "max": int(per_pair.max()),
            },
        }

    return metrics


def rates_at(true_losses, false_losses, threshold):
    """FN and FP percentages at one threshold: accept a pair when L* <= T."""
    return {
        "threshold": float(threshold),
        "false_negative_percent": 100.0 * float(np.mean(true_losses > threshold)),
        "false_positive_percent": 100.0 * float(np.mean(false_losses <= threshold)),
    }


def best_threshold(true_losses, false_losses):
    """The candidate threshold with the fewest errors, the smallest if tied."""
    candidates = np.sort(np.concatenate((true_losses, false_losses)))
    errors = [
        int(np.sum(true_losses > t) + np.sum(false_losses <= t)) for t in candidates
    ]
    return float(candidates[int(np.argmin(errors))])


def decision_metrics(output_dir):
    """Metrics 7 and 8: how well one threshold would separate the pairs.

    Nothing applies a threshold yet, so this measures what one could achieve.
    best_threshold is chosen on the same pairs it is scored on, which makes its
    rates optimistic.
    """
    metrics = {}

    for variant in VARIANTS:
        with np.load(output_dir / f"association_{variant}.npz") as archive:
            is_true, loss = archive["is_true"], archive["loss"]

        true_losses, false_losses = loss[is_true], loss[~is_true]
        worst_true, best_false = float(true_losses.max()), float(false_losses.min())

        metrics[variant] = {
            "pairs": {"true": len(true_losses), "false": len(false_losses)},
            "best_threshold": rates_at(
                true_losses, false_losses, best_threshold(true_losses, false_losses),
            ),
            "separation_gap": {
                "worst_true": worst_true,
                "best_false": best_false,
                "ratio": best_false / worst_true,
                "true_pairs_at_or_above_best_false": int(np.sum(true_losses >= best_false)),
            },
        }
        if variant != "clean":
            metrics[variant]["paper_thresholds"] = [
                rates_at(true_losses, false_losses, t) for t in PAPER_THRESHOLDS
            ]

    return metrics
