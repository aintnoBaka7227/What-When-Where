"""Tests for stage 8, optimise_hypothesis.py.

    uv run python test/test_search.py

The black box is scipy's L-BFGS-B. Tested here: the Eq. (31) seed, the
infeasibility penalty, the gradient step, and the search on generated data.
Two known limits, a flat penalty plateau and a second basin in some
multi-revolution landscapes, are measured and bounded. See TESTING.md.
"""

from itertools import combinations

import numpy as np

from lab.evaluation.metrics import true_k
from lab.hypothesis.k_bounds import orbital_period
from lab.hypothesis.range_bounds import calculate_u_i, population_for
from lab.model.lambert import LambertNoSolution
from lab.scoring.discriminator_loss import evaluate_loss
from lab.search.optimise_hypothesis import (
    MU,
    _INFEASIBLE_PENALTY,
    _MINIMIZER_OPTIONS,
    _objective,
    best_k_hypothesis,
    circular_orbit_initial_guess,
    optimize_hypothesis,
    semi_major_axis_for_k,
)
from pipeline_fixture import pipeline, run_tests

A_MIN, A_MAX = 38000.0, 45000.0


def _a_for_revolutions(delta_t, revolutions):
    period = delta_t / revolutions
    return (MU * period**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)


# --------------------------------------------------------------------------
# seed: simple cases
# --------------------------------------------------------------------------

def test_seed_is_the_bin_midpoint_when_the_bin_is_admissible():
    """3 days spans 2.73-3.52 revolutions over the population, so k = 3 fits whole."""
    delta_t = 3 * 86400.0
    assert delta_t / orbital_period(A_MAX) < 3.0 and delta_t / orbital_period(A_MIN) > 3.5

    assert np.isclose(semi_major_axis_for_k(3.0, delta_t), _a_for_revolutions(delta_t, 3.25), rtol=1e-12)
    assert np.isclose(semi_major_axis_for_k(2.5, delta_t),
                      _a_for_revolutions(delta_t, 0.5 * (delta_t / orbital_period(A_MAX) + 3.0)),
                      rtol=1e-12)


def test_seed_is_in_the_population_and_in_its_bin():
    rng = np.random.default_rng(4)
    for _ in range(300):
        delta_t = rng.uniform(2_000.0, 300_000.0)
        k_low = np.floor(2 * delta_t / orbital_period(A_MAX)) / 2
        k_high = np.floor(2 * delta_t / orbital_period(A_MIN)) / 2
        for k in np.arange(k_low, k_high + 0.25, 0.5):
            a = semi_major_axis_for_k(k, delta_t)
            revolutions = delta_t / orbital_period(a)
            assert A_MIN <= a <= A_MAX
            assert k <= revolutions < k + 0.5, (delta_t, k, revolutions)


def test_seed_keeps_away_from_degenerate_transfer_angles():
    """A wholly admissible bin seeds a 90 deg transfer, not the old near-0 deg edge."""
    delta_t = 3 * 86400.0
    for k in (3.0,):
        revolutions = delta_t / orbital_period(semi_major_axis_for_k(k, delta_t))
        sweep_deg = 360.0 * (revolutions % 1.0)
        assert np.isclose(sweep_deg, 90.0), sweep_deg


def test_circular_guess_from_the_geocentre_is_a_k():
    delta_t = 3 * 86400.0
    a = semi_major_axis_for_k(3.0, delta_t)
    rho_1, rho_2 = circular_orbit_initial_guess(3.0, delta_t, 0.0, np.zeros(3), 0.0, np.zeros(3))
    assert np.isclose(rho_1, a) and np.isclose(rho_2, a)


def test_circular_guess_lands_on_the_radius_a_k():
    """Eq. 31 is Eq. 26 with |r| = a(k): r_s + rho* u lies on that sphere."""
    delta_t = 3 * 86400.0
    a = semi_major_axis_for_k(3.0, delta_t)
    r_s = np.array([4000.0, 3000.0, 3800.0])
    u = calculate_u_i(0.9, 0.4)
    c = float(r_s @ u)
    rho, _ = circular_orbit_initial_guess(3.0, delta_t, c, r_s, c, r_s)
    assert np.isclose(np.linalg.norm(r_s + rho * u), a, rtol=1e-12)


# --------------------------------------------------------------------------
# seed and objective: edge cases
# --------------------------------------------------------------------------

def test_zero_interval_raises():
    try:
        semi_major_axis_for_k(0.0, 0.0)
    except ValueError:
        return
    raise AssertionError("dt = 0 produced a seed")


def test_unreachable_k_raises():
    """One day cannot hold 5 revolutions of a near-GEO orbit."""
    try:
        semi_major_axis_for_k(5.0, 86400.0)
    except ValueError:
        return
    raise AssertionError("an unreachable k produced a seed")


def test_infeasible_hypothesis_scores_the_penalty():
    """k = 1 over 2 h is below the one-revolution minimum time of flight."""
    k, rho = 1.0, 36000.0
    r_s, v_s, ca = np.array([6378.0, 0.0, 0.0]), np.zeros(3), np.zeros((4, 4))
    tracklet_1 = (r_s, v_s, 0.0, np.array([0.0, 7e-5, 0.0, 0.0]), ca)
    tracklet_2 = (r_s, v_s, 7200.0, np.array([0.5, 7e-5, 0.0, 0.0]), ca)

    try:
        evaluate_loss("clean", rho, rho, k, *tracklet_1, *tracklet_2)
    except LambertNoSolution:
        pass
    else:
        raise AssertionError("test geometry is feasible; pick another")

    value = _objective([rho, rho], "clean", k, *tracklet_1, *tracklet_2, MU, 1e-6)
    assert value == _INFEASIBLE_PENALTY


def test_gradient_step_is_resolvable():
    """eps must be far above rounding at 4e4 km (~1e-11 km) and far below the
    ~500 km width of the loss valley."""
    assert 1e-6 < _MINIMIZER_OPTIONS["eps"] < 10.0


# --------------------------------------------------------------------------
# on the generator's data
# --------------------------------------------------------------------------

def _true_pair_searches(run, variant):
    """Search every true pair at its true k and over its k grid.

    Yields a dict per pair: truth, the true-k result, the grid result.
    """
    a_min, a_max, _ = population_for(variant)
    attributables = run.attributables(variant)
    bounds = run.range_bounds(variant)
    pairs = run.k_bounds(variant)
    ids = list(zip(attributables["object_id"], attributables["tracklet_id"]))

    for p in range(len(pairs["k_min"])):
        if pairs["object_id_1"][p] != pairs["object_id_2"][p]:
            continue
        i = ids.index((pairs["object_id_1"][p], pairs["tracklet_id_1"][p]))
        j = ids.index((pairs["object_id_2"][p], pairs["tracklet_id_2"][p]))
        t1, r1, v1, r_s1, v_s1 = run.truth_at_epoch(*ids[i])
        t2, r2, _, r_s2, v_s2 = run.truth_at_epoch(*ids[j])
        rho_true = (np.linalg.norm(r1 - r_s1), np.linalg.norm(r2 - r_s2))
        k = true_k(r1, v1, r2, t2 - t1)

        tracklet_1 = (bounds["c_i_km"][i], r_s1, v_s1, t1,
                      (bounds["rho_min_i_km"][i], bounds["rho_max_i_km"][i]),
                      attributables["attributable"][i], attributables["Ca"][i])
        tracklet_2 = (bounds["c_i_km"][j], r_s2, v_s2, t2,
                      (bounds["rho_min_i_km"][j], bounds["rho_max_i_km"][j]),
                      attributables["attributable"][j], attributables["Ca"][j])

        loss_true = min(evaluate_loss(
            variant, *rho_true, k,
            r_s1, v_s1, t1, tracklet_1[5], tracklet_1[6],
            r_s2, v_s2, t2, tracklet_2[5], tracklet_2[6],
        ))
        rho_1, rho_2, loss, _ = optimize_hypothesis(
            variant, k, t2 - t1, *tracklet_1, *tracklet_2, a_min=a_min, a_max=a_max,
        )
        k_best, *_ = best_k_hypothesis(
            variant, pairs["k_min"][p], pairs["k_max"][p], t2 - t1,
            *tracklet_1, *tracklet_2, a_min=a_min, a_max=a_max,
        )

        # Highest loss on the straight line from the answer to the truth.
        objective_args = (variant, k, r_s1, v_s1, t1, tracklet_1[5], tracklet_1[6],
                          r_s2, v_s2, t2, tracklet_2[5], tracklet_2[6], MU, 1e-6)
        barrier = max(
            _objective(np.array([rho_1, rho_2]) * (1 - s) + np.array(rho_true) * s, *objective_args)
            for s in np.linspace(0.0, 1.0, 21)
        )
        yield {
            "k": k, "k_best": k_best, "rho_true": rho_true, "rho": (rho_1, rho_2),
            "loss": loss, "loss_true": loss_true, "barrier": barrier,
        }


def test_clean_search_reaches_the_true_ranges():
    with pipeline(num_objects=4, stages=4) as run:
        results = list(_true_pair_searches(run, "clean"))

    reached = [r for r in results if r["loss"] < _INFEASIBLE_PENALTY]
    errors = np.array([np.subtract(r["rho"], r["rho_true"]) for r in reached])
    assert len(reached) >= 0.85 * len(results), (len(reached), len(results))
    # Clean data: the minimum is the truth; 50 km is far inside the ~500 km valley.
    assert np.median(np.abs(errors)) < 5.0, np.median(np.abs(errors))
    assert np.mean(np.abs(errors).max(axis=1) < 50.0) > 0.9


def test_noisy_search_scores_no_worse_than_the_truth():
    """Ending above the truth's loss is a bug (an early stop) or a second basin.

    Each such case needs a real barrier on the line to the truth, and they must
    stay rare. Here there is one: a k = 1.5 pair where the loss rises from 2.3e3
    to 4.5e3 before falling to 4.3 at the truth.
    """
    with pipeline(num_objects=4, stages=4) as run:
        results = list(_true_pair_searches(run, "noisy_0.7as"))

    ran = [r for r in results if r["loss"] < _INFEASIBLE_PENALTY]
    worse = [r for r in ran if r["loss"] > r["loss_true"] * (1 + 1e-6) + 1e-9]

    assert len(worse) <= 1, [(r["k"], r["loss"], r["loss_true"]) for r in worse]
    for r in worse:
        assert r["k"] >= 1.0, "a single-revolution search ended in a second basin"
        assert r["barrier"] > r["loss"], (
            f"no barrier between L* = {r['loss']:.3g} and the truth: the search stopped early"
        )


def test_penalty_starts_at_the_true_k_stay_rare():
    """The known limit. With the bin-midpoint seed, 3 of 40 true-k searches
    on this dataset start and stay on the penalty; the old lower-edge seed
    gave 7. Fail if it gets worse."""
    with pipeline(num_objects=4, stages=4) as run:
        results = list(_true_pair_searches(run, "clean"))
    stuck = sum(r["loss"] >= _INFEASIBLE_PENALTY for r in results)
    assert len(results) == 40
    assert stuck <= 3, f"{stuck} of {len(results)} true-k searches stayed on the penalty"


def test_search_over_the_grid_picks_the_true_k():
    with pipeline(num_objects=4, stages=4) as run:
        for variant in ("clean", "noisy_0.7as"):
            results = list(_true_pair_searches(run, variant))
            correct = sum(r["k_best"] == r["k"] for r in results)
            # 38 of 40 here; 97 of 100 on the 10-object run.
            assert correct >= 37, f"{variant}: true k chosen for {correct} of {len(results)}"


TESTS = [
    test_seed_is_the_bin_midpoint_when_the_bin_is_admissible,
    test_seed_is_in_the_population_and_in_its_bin,
    test_seed_keeps_away_from_degenerate_transfer_angles,
    test_circular_guess_from_the_geocentre_is_a_k,
    test_circular_guess_lands_on_the_radius_a_k,
    test_zero_interval_raises,
    test_unreachable_k_raises,
    test_infeasible_hypothesis_scores_the_penalty,
    test_gradient_step_is_resolvable,
    test_clean_search_reaches_the_true_ranges,
    test_noisy_search_scores_no_worse_than_the_truth,
    test_penalty_starts_at_the_true_k_stay_rare,
    test_search_over_the_grid_picks_the_true_k,
]


if __name__ == "__main__":
    raise SystemExit(run_tests(TESTS))
