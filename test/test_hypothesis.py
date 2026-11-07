"""Tests for stages 3 and 4, range_bounds.py and k_bounds.py.

    uv run python test/test_hypothesis.py

Closed-form equations, so: hand-worked cases, edge cases, the defining
property |r_s + rho u| = |r|, then generated data. See TESTING.md.
"""

from itertools import combinations

import numpy as np

from lab.data.dataset import SITE_POSITION, TIME_COLUMN, VARIANTS, load_data
from lab.evaluation.metrics import true_k
from lab.hypothesis.k_bounds import MU, calculate_k_bounds, k_grid, orbital_period
from lab.hypothesis.range_bounds import (
    calculate_r_min_r_max,
    calculate_rho_bounds_i,
    calculate_u_i,
    r_s_i_at_epoch,
)
from pipeline_fixture import pipeline, run_tests

R_EARTH_KM = 6378.137
SIDEREAL_DAY_S = 86164.0905
GEO_RADIUS_KM = 42164.17


# --------------------------------------------------------------------------
# stage 3: simple cases
# --------------------------------------------------------------------------

def test_r_min_r_max_equations_23_24():
    r_min, r_max = calculate_r_min_r_max(38000.0, 45000.0, 0.01)
    assert np.isclose(r_min, 37620.0) and np.isclose(r_max, 45450.0)


def test_line_of_sight_directions():
    assert np.allclose(calculate_u_i(0.0, 0.0), [1, 0, 0])
    assert np.allclose(calculate_u_i(np.pi / 2, 0.0), [0, 1, 0], atol=1e-15)
    assert np.allclose(calculate_u_i(1.234, np.pi / 2), [0, 0, 1], atol=1e-15)
    # RA is periodic: 2*pi + x must give the same direction as x.
    assert np.allclose(calculate_u_i(2 * np.pi + 0.3, 0.1), calculate_u_i(0.3, 0.1))
    for alpha, delta in [(0.4, -0.3), (5.9, 0.2), (3.1, -1.5)]:
        assert np.isclose(np.linalg.norm(calculate_u_i(alpha, delta)), 1.0)


def test_site_at_the_geocentre_gives_rho_equal_to_radius():
    c, rho_min, rho_max = calculate_rho_bounds_i(np.zeros(3), calculate_u_i(0.7, 0.2), 37620.0, 45450.0)
    assert c == 0.0
    assert np.isclose(rho_min, 37620.0) and np.isclose(rho_max, 45450.0)


def test_zenith_line_of_sight_subtracts_the_site_radius():
    r_s = np.array([0.0, 0.0, R_EARTH_KM])
    _, rho_min, rho_max = calculate_rho_bounds_i(r_s, np.array([0.0, 0.0, 1.0]), 37620.0, 45450.0)
    assert np.isclose(rho_min, 37620.0 - R_EARTH_KM)
    assert np.isclose(rho_max, 45450.0 - R_EARTH_KM)


def test_horizontal_line_of_sight_is_pythagoras():
    r_s = np.array([R_EARTH_KM, 0.0, 0.0])
    c, rho_min, _ = calculate_rho_bounds_i(r_s, np.array([0.0, 1.0, 0.0]), 37620.0, 45450.0)
    assert c == 0.0
    assert np.isclose(rho_min, np.sqrt(37620.0**2 - R_EARTH_KM**2))


def test_bounds_land_exactly_on_the_shells():
    """The defining property, over random sites and lines of sight above the horizon."""
    rng = np.random.default_rng(1)
    r_min, r_max = 37620.0, 45450.0
    for _ in range(200):
        r_s = rng.normal(size=3)
        r_s *= R_EARTH_KM / np.linalg.norm(r_s)
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        if u @ r_s < 0:
            u = -u
        _, rho_min, rho_max = calculate_rho_bounds_i(r_s, u, r_min, r_max)

        assert 0.0 < rho_min < rho_max
        assert np.isclose(np.linalg.norm(r_s + rho_min * u), r_min, rtol=1e-12)
        assert np.isclose(np.linalg.norm(r_s + rho_max * u), r_max, rtol=1e-12)


# --------------------------------------------------------------------------
# stage 3: edge cases
# --------------------------------------------------------------------------

def test_single_orbit_radius_collapses_the_bounds():
    r_min, r_max = calculate_r_min_r_max(42164.0, 42164.0, 0.0)
    _, rho_min, rho_max = calculate_rho_bounds_i(
        np.array([R_EARTH_KM, 0, 0]), calculate_u_i(0.3, 0.1), r_min, r_max,
    )
    assert rho_min == rho_max


def test_larger_eccentricity_widens_both_bounds():
    r_s = np.array([R_EARTH_KM, 0, 0])
    u = calculate_u_i(0.3, 0.1)
    _, low_a, high_a = calculate_rho_bounds_i(r_s, u, *calculate_r_min_r_max(38000, 45000, 0.01))
    _, low_b, high_b = calculate_rho_bounds_i(r_s, u, *calculate_r_min_r_max(38000, 45000, 0.2))
    assert low_b < low_a and high_b > high_a


def test_site_interpolation_at_and_between_samples():
    rows = np.zeros((3, 18))
    rows[:, TIME_COLUMN] = [0.0, 20.0, 40.0]
    rows[:, SITE_POSITION] = [[0, 0, 0], [10, 20, 30], [20, 40, 60]]
    assert np.array_equal(r_s_i_at_epoch(rows, 0, 0, 20.0), [10, 20, 30])
    assert np.allclose(r_s_i_at_epoch(rows, 0, 0, 30.0), [15, 30, 45])


# --------------------------------------------------------------------------
# stage 3: on the generator's data
# --------------------------------------------------------------------------

def _true_ranges_inside(run, variant):
    bounds = run.range_bounds(variant)
    margins = []
    for n, (obj, trk) in enumerate(zip(bounds["object_id"], bounds["tracklet_id"])):
        _, r, _, r_site, _ = run.truth_at_epoch(obj, trk)
        rho = np.linalg.norm(r - r_site)
        margins.append(min(rho - bounds["rho_min_i_km"][n], bounds["rho_max_i_km"][n] - rho))
    return np.array(margins)


def test_true_range_is_inside_the_bounds_in_every_chain():
    with pipeline(num_objects=6, stages=3) as run:
        for variant in VARIANTS:
            margins = _true_ranges_inside(run, variant)
            assert np.all(margins > 0.0), (variant, margins.min())


def test_bounds_follow_an_eccentric_population():
    """Test case 'b' (e up to 0.2) would fall outside e_max = 0.01 bounds."""
    with pipeline(num_objects=4, stages=3, test_case="b") as run:
        bounds = run.range_bounds("clean")
        assert float(bounds["e_max"]) == 0.2
        assert np.all(_true_ranges_inside(run, "clean") > 0.0)


def test_site_at_epoch_is_the_middle_observation():
    """t0 is the mean time of evenly spaced samples, so no interpolation error."""
    with pipeline(num_objects=2, stages=3) as run:
        bounds = run.range_bounds("clean")
        for n, (obj, trk) in enumerate(zip(bounds["object_id"], bounds["tracklet_id"])):
            _, _, _, r_site, _ = run.truth_at_epoch(obj, trk)
            assert np.allclose(bounds["r_s_i_km"][n], r_site, rtol=0.0, atol=1e-9)


def test_chains_differ_only_through_the_line_of_sight():
    with pipeline(num_objects=2, stages=3) as run:
        clean = run.range_bounds("clean")
        for variant in VARIANTS[1:]:
            noisy = run.range_bounds(variant)
            assert np.array_equal(noisy["r_s_i_km"], clean["r_s_i_km"])
            assert np.array_equal(noisy["t0"], clean["t0"])
            assert not np.array_equal(noisy["u_i"], clean["u_i"])


# --------------------------------------------------------------------------
# stage 4: simple cases
# --------------------------------------------------------------------------

def test_geo_period_is_a_sidereal_day():
    assert abs(orbital_period(GEO_RADIUS_KM) - SIDEREAL_DAY_S) < 1.0


def test_k_counts_half_revolutions():
    a = 42164.0
    period = orbital_period(a)
    for fraction, k in [(0.25, 0.0), (0.75, 0.5), (1.25, 1.0), (1.75, 1.5), (2.1, 2.0)]:
        assert calculate_k_bounds(fraction * period, a, a) == (k, k), fraction


def test_k_grid_enumerates_half_steps():
    assert k_grid(0.0, 0.0) == [0.0]
    assert k_grid(0.5, 1.5) == [0.5, 1.0, 1.5]
    assert k_grid(2.0, 3.0) == [2.0, 2.5, 3.0]


# --------------------------------------------------------------------------
# stage 4: edge cases
# --------------------------------------------------------------------------

def test_zero_interval_gives_k_zero():
    assert calculate_k_bounds(0.0, 38000.0, 45000.0) == (0.0, 0.0)


def test_exactly_half_a_period_rounds_up_to_the_next_half():
    a = 42164.0
    period = orbital_period(a)
    assert calculate_k_bounds(0.5 * period * (1 - 1e-12), a, a) == (0.0, 0.0)
    assert calculate_k_bounds(0.5 * period * (1 + 1e-12), a, a) == (0.5, 0.5)


def test_sign_of_the_interval_does_not_matter():
    assert calculate_k_bounds(-100_000.0, 38000.0, 45000.0) == calculate_k_bounds(100_000.0, 38000.0, 45000.0)


def test_k_min_is_never_above_k_max():
    """The paper's labels are reversed: a_min gives the larger raw value."""
    for dt in np.linspace(1_000.0, 300_000.0, 300):
        k_min, k_max = calculate_k_bounds(dt, 38000.0, 45000.0)
        assert k_min <= k_max
        raw_at_a_min = np.floor(2 * dt / orbital_period(38000.0)) / 2
        assert k_max == raw_at_a_min


def test_true_revolutions_fall_in_the_grid_for_any_admissible_a():
    rng = np.random.default_rng(2)
    for _ in range(500):
        a = rng.uniform(38000.0, 45000.0)
        dt = rng.uniform(1_000.0, 300_000.0)
        k_true = np.floor(2 * dt / orbital_period(a)) / 2
        k_min, k_max = calculate_k_bounds(dt, 38000.0, 45000.0)
        assert k_min <= k_true <= k_max, (a, dt)


# --------------------------------------------------------------------------
# stage 4: on the generator's data
# --------------------------------------------------------------------------

def test_every_tracklet_pair_is_kept():
    """Per-object start offsets mean no shared epochs, so no pair is dropped."""
    with pipeline(num_objects=5, stages=4) as run:
        t0 = run.attributables("clean")["t0"]
        pairs = run.k_bounds("clean")
        assert len(pairs["k_min"]) == len(t0) * (len(t0) - 1) // 2
        assert np.all(pairs["delta_t_s"] > 0.0)


def test_delta_t_is_the_epoch_separation():
    with pipeline(num_objects=3, stages=4) as run:
        attributables = run.attributables("clean")
        epoch = dict(zip(zip(attributables["object_id"], attributables["tracklet_id"]), attributables["t0"]))
        pairs = run.k_bounds("clean")
        for o1, t1, o2, t2, dt in zip(pairs["object_id_1"], pairs["tracklet_id_1"],
                                      pairs["object_id_2"], pairs["tracklet_id_2"], pairs["delta_t_s"]):
            assert dt == abs(epoch[(o2, t2)] - epoch[(o1, t1)])


def test_true_k_is_in_the_grid_for_every_true_pair():
    with pipeline(num_objects=6, stages=4) as run:
        pairs = run.k_bounds("clean")
        checked = 0
        for o1, t1, o2, t2, k_min, k_max in zip(
            pairs["object_id_1"], pairs["tracklet_id_1"], pairs["object_id_2"],
            pairs["tracklet_id_2"], pairs["k_min"], pairs["k_max"],
        ):
            if o1 != o2:
                continue
            e1, r1, v1, _, _ = run.truth_at_epoch(o1, t1)
            e2, r2, _, _, _ = run.truth_at_epoch(o2, t2)
            k = true_k(r1, v1, r2, e2 - e1)
            assert k_min <= k <= k_max, (o1, t1, t2, k, k_min, k_max)
            checked += 1
        assert checked == 6 * 10


def test_every_chain_writes_the_same_k_bounds():
    with pipeline(num_objects=2, stages=4) as run:
        clean = run.k_bounds("clean")
        for variant in VARIANTS[1:]:
            other = run.k_bounds(variant)
            for key in clean:
                assert np.array_equal(other[key], clean[key]), (variant, key)


TESTS = [
    test_r_min_r_max_equations_23_24,
    test_line_of_sight_directions,
    test_site_at_the_geocentre_gives_rho_equal_to_radius,
    test_zenith_line_of_sight_subtracts_the_site_radius,
    test_horizontal_line_of_sight_is_pythagoras,
    test_bounds_land_exactly_on_the_shells,
    test_single_orbit_radius_collapses_the_bounds,
    test_larger_eccentricity_widens_both_bounds,
    test_site_interpolation_at_and_between_samples,
    test_true_range_is_inside_the_bounds_in_every_chain,
    test_bounds_follow_an_eccentric_population,
    test_site_at_epoch_is_the_middle_observation,
    test_chains_differ_only_through_the_line_of_sight,
    test_geo_period_is_a_sidereal_day,
    test_k_counts_half_revolutions,
    test_k_grid_enumerates_half_steps,
    test_zero_interval_gives_k_zero,
    test_exactly_half_a_period_rounds_up_to_the_next_half,
    test_sign_of_the_interval_does_not_matter,
    test_k_min_is_never_above_k_max,
    test_true_revolutions_fall_in_the_grid_for_any_admissible_a,
    test_every_tracklet_pair_is_kept,
    test_delta_t_is_the_epoch_separation,
    test_true_k_is_in_the_grid_for_every_true_pair,
    test_every_chain_writes_the_same_k_bounds,
]


if __name__ == "__main__":
    raise SystemExit(run_tests(TESTS))
