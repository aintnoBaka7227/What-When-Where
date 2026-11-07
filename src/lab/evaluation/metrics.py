"""Metrics for judging each pipeline stage against the simulated truth.

Each function reads a run's output folder and returns a plain dict, which
experiment/run/ saves as metrics.json in that run's folder:

    data_metrics         is the dataset what it claims to be, and realistic?
    hypothesis_metrics   is the truth inside the admissible region? (Eqs. 22, 29)
    scoring_metrics      does the loss score the truth as a match? (Eq. 18)
    search_metrics       did the optimiser reach the right answer? (Eq. 30)
    decision_metrics     how well does a threshold separate pairs? (Fig. 9)

Truth comes from columns 0-5 of the data matrix, which only metrics may read.
Those columns are the same in every chain, so they are read from clean.
"""

import json
from itertools import combinations

import numpy as np
from astropy.constants import GM_earth
from scipy import stats

from lab.data.dataset import (
    DEC_COLUMN,
    NOISY_VARIANTS,
    OBJECT_ID_COLUMN,
    RA_COLUMN,
    SATELLITE_POSITION,
    SATELLITE_VELOCITY,
    SITE_POSITION,
    TIME_COLUMN,
    TRACKLET_ID_COLUMN,
    VARIANTS,
)

# km^3/s^2, as in the generator.
MU = GM_earth.value / 1_000_000_000.0

# optimise_hypothesis._INFEASIBLE_PENALTY.
INFEASIBLE_PENALTY = 1.0e12

ARCSEC = np.deg2rad(1.0 / 3600.0)

# Earth's rotation rate, the sky motion of a geostationary object.
GEO_RATE_ARCSEC_S = 15.04

# A pair within this of 0 or 180 degrees counts as near-degenerate.
NEAR_DEGENERATE_DEG = 5.0

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
            (self.rows[:, OBJECT_ID_COLUMN] == object_id)
            & (self.rows[:, TRACKLET_ID_COLUMN] == tracklet_id)
            & np.isclose(self.rows[:, TIME_COLUMN], t0)
        )
        return self.rows[match][0]

    def range_km(self, object_id, tracklet_id):
        row = self.row(object_id, tracklet_id)
        return float(np.linalg.norm(row[SATELLITE_POSITION] - row[SITE_POSITION]))

    def k(self, object_id, tracklet_1, tracklet_2):
        row_1 = self.row(object_id, tracklet_1)
        row_2 = self.row(object_id, tracklet_2)
        if row_2[TIME_COLUMN] < row_1[TIME_COLUMN]:
            row_1, row_2 = row_2, row_1
        return true_k(
            row_1[SATELLITE_POSITION], row_1[SATELLITE_VELOCITY],
            row_2[SATELLITE_POSITION], row_2[TIME_COLUMN] - row_1[TIME_COLUMN],
        )


def share(count, total):
    return float(count / total) if total else None


def uniform_fit(values, low, high):
    """Kolmogorov-Smirnov test of values against a uniform distribution."""
    result = stats.kstest(values, "uniform", args=(low, high - low))
    return {
        "n": len(values),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "expected_range": [float(low), float(high)],
        "ks_p_value": float(result.pvalue),
    }


def normal_fit(values, sigma):
    """Kolmogorov-Smirnov test of values against a zero-mean normal distribution."""
    result = stats.kstest(values, "norm", args=(0.0, sigma))
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "expected_sd": float(sigma),
        "ks_p_value": float(result.pvalue),
    }


def orbit_elements(state):
    """(a, e, inclination in degrees) of a Cartesian state, without the generator."""
    r, v = state[SATELLITE_POSITION], state[SATELLITE_VELOCITY]
    r_mag = np.linalg.norm(r)
    h = np.cross(r, v)
    a = 1.0 / (2.0 / r_mag - np.dot(v, v) / MU)
    e = np.linalg.norm(np.cross(v, h) / MU - r / r_mag)
    inclination = np.degrees(np.arccos(h[2] / np.linalg.norm(h)))
    return a, e, inclination


def on_sky_rate_arcsec_s(tracklet_rows):
    """Straight-line fit to a tracklet's clean RA/Dec, as motion across the sky."""
    times = tracklet_rows[:, TIME_COLUMN]
    ra_rate = np.polyfit(times, np.unwrap(tracklet_rows[:, RA_COLUMN]), 1)[0]
    dec_rate = np.polyfit(times, tracklet_rows[:, DEC_COLUMN], 1)[0]
    cos_dec = np.cos(tracklet_rows[:, DEC_COLUMN].mean())
    return float(np.hypot(ra_rate * cos_dec, dec_rate)) / ARCSEC


def data_metrics(output_dir):
    """Layers 2 and 3 of validating stage 1: statistics and realism.

    Each drawn quantity is tested against the distribution it should follow,
    recomputed from the rows rather than from the generator's own draws where
    possible. A KS p-value below about 0.01 means the draws do not fit. With few
    objects these tests have little power, so judge them on 100 or more.
    """
    with (output_dir / "generated_data_gcrs_settings.json").open() as handle:
        record = json.load(handle)
    settings, objects = record["settings"], record["objects"]
    rows = np.loadtxt(output_dir / "generated_data_gcrs_clean.txt", delimiter=",", ndmin=2)

    per_tracklet = settings["num_of_observations_per_tracklet"]
    first_rows = rows[::per_tracklet]
    elements = np.array([
        orbit_elements(rows[rows[:, OBJECT_ID_COLUMN] == obj["object_id"]][0]) for obj in objects
    ])

    # Population, from the rows.
    population = {
        "semi_major_axis_km": uniform_fit(elements[:, 0], *settings["semi_major_axis_range_km"]),
        "eccentricity": uniform_fit(elements[:, 1], *settings["eccentricity_range"]),
        "inclination_deg": uniform_fit(elements[:, 2], *settings["inclination_range_deg"]),
    }

    # Stations: the true offset at each tracklet's first observation.
    satellite, site = first_rows[:, SATELLITE_POSITION], first_rows[:, SITE_POSITION]
    cosine = np.sum(satellite * site, axis=1) / (
        np.linalg.norm(satellite, axis=1) * np.linalg.norm(site, axis=1)
    )
    offsets = np.degrees(np.arccos(cosine))

    line_of_sight = rows[:, SATELLITE_POSITION] - rows[:, SITE_POSITION]
    up = rows[:, SITE_POSITION] / np.linalg.norm(rows[:, SITE_POSITION], axis=1, keepdims=True)
    elevation = np.degrees(np.arcsin(
        np.sum(up * line_of_sight, axis=1) / np.linalg.norm(line_of_sight, axis=1)
    ))

    # Timing, relative to the schedule.
    starts_h = first_rows[:, TIME_COLUMN].reshape(len(objects), -1) / 3600.0
    schedule = np.array(settings["tracklet_schedule_h"])
    jitter = (starts_h - starts_h[:, :1] - schedule)[:, 1:].ravel()
    max_jitter = settings["max_tracklet_jitter_h"]

    # Noise: each chain's residual once the recorded bias is removed.
    biases = np.array([
        (t["ra_systematic_bias_arcsec"], t["dec_systematic_bias_arcsec"])
        for obj in objects for t in obj["tracklets"]
    ])
    noise = {}
    for variant, sigma in NOISY_VARIANTS.items():
        noisy = np.loadtxt(
            output_dir / f"generated_data_gcrs_{variant}.txt", delimiter=",", ndmin=2,
        )
        difference = np.column_stack((
            (noisy[:, RA_COLUMN] - rows[:, RA_COLUMN] + np.pi) % (2.0 * np.pi) - np.pi,
            noisy[:, DEC_COLUMN] - rows[:, DEC_COLUMN],
        )) / ARCSEC
        residual = difference - np.repeat(biases, per_tracklet, axis=0)
        noise[variant] = normal_fit(residual.ravel(), sigma)

    # Geometry of the pairs.
    near_degenerate = []
    for obj in objects:
        positions = first_rows[first_rows[:, OBJECT_ID_COLUMN] == obj["object_id"]][:, SATELLITE_POSITION]
        for r1, r2 in combinations(positions, 2):
            angle = np.degrees(np.arccos(np.clip(
                np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)), -1.0, 1.0,
            )))
            near_degenerate.append(min(angle, 180.0 - angle) < NEAR_DEGENERATE_DEG)

    # Realism against the paper.
    rates = np.array([
        on_sky_rate_arcsec_s(rows[rows[:, TRACKLET_ID_COLUMN] == tracklet])
        for tracklet in np.unique(rows[:, TRACKLET_ID_COLUMN])
    ])
    duration = settings["tracklet_duration_s"]

    return {
        "statistics": {
            "population": population,
            "station_offset_deg": uniform_fit(
                offsets, 0.0, settings["station_max_offset_from_subsatellite_deg"],
            ),
            "elevation_deg": {"min": float(elevation.min()), "max": float(elevation.max())},
            "start_offset_h": uniform_fit(starts_h[:, 0], 0.0, settings["max_start_offset_h"]),
            "tracklet_jitter_h": uniform_fit(jitter, -max_jitter, max_jitter),
            "systematic_bias_arcsec": normal_fit(
                biases.ravel(), settings["systematic_noise_arcsec"],
            ),
            "relative_noise_arcsec": noise,
            "near_degenerate_pairs": {
                "pairs": len(near_degenerate),
                "share": share(sum(near_degenerate), len(near_degenerate)),
            },
        },
        "realism": {
            "on_sky_rate_arcsec_s": {
                "median": float(np.median(rates)),
                "min": float(rates.min()),
                "max": float(rates.max()),
                "geostationary_reference": GEO_RATE_ARCSEC_S,
            },
            "tracklet_arc_arcmin": float(np.median(rates) * duration / 60.0),
            "paper_comparison": {
                "tracklet_duration_s": {"this_dataset": duration, "paper": "60-120"},
                "observations_per_tracklet": {"this_dataset": per_tracklet, "paper": "3-5"},
                "observation_spacing_s": {
                    "this_dataset": settings["time_between_observations_s"],
                    "paper": "around 20",
                },
                "relative_noise_arcsec": {
                    "this_dataset": settings["relative_noise_levels_arcsec"],
                    "paper": [0.7, 1.2, 2.0],
                },
                "systematic_noise_arcsec": {
                    "this_dataset": settings["systematic_noise_arcsec"],
                    "paper": 5.0,
                },
                "tracklets_per_object": {
                    "this_dataset": settings["num_of_tracklets_per_object"],
                    "paper": "1-2",
                },
                "time_span_h": {
                    "this_dataset": float(first_rows[:, TIME_COLUMN].max() / 3600.0),
                    "paper": "about 27",
                },
            },
        },
    }


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
