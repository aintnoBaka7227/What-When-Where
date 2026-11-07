"""Tests for stages 5 and 6, build_orbit.py and model_rates.py.

    uv run python test/test_model.py

The Lambert solver itself is in test_lambert.py. Here: Eq. (5) and tracklet
ordering, recovery of the true orbit from generated tracklets, and the Eq. (17)
derivative by hand cases, finite differences and generated data. See TESTING.md.
"""

from itertools import combinations

import numpy as np

from lab.data.gcrs_data_generation import two_body
from lab.evaluation.metrics import true_k
from lab.model.build_orbit import MU, build_orbit, position_from_range
from lab.model.lambert import LambertNoSolution
from lab.model.model_rates import angle_rate_from_state, modeled_angular_rates
from pipeline_fixture import pipeline, run_tests

ARCSEC = np.deg2rad(1.0 / 3600.0)


def _angles(rho):
    return np.arctan2(rho[1], rho[0]), np.arcsin(rho[2] / np.linalg.norm(rho))


def _same_object_pairs(run):
    """(tracklet 1, tracklet 2) for every same-object pair, with truth at t0."""
    attributables = run.attributables("clean")
    ids = list(zip(attributables["object_id"], attributables["tracklet_id"]))
    for (o1, t1), (o2, t2) in combinations(ids, 2):
        if o1 == o2:
            yield run.truth_at_epoch(o1, t1), run.truth_at_epoch(o2, t2)


# --------------------------------------------------------------------------
# stage 5: simple cases
# --------------------------------------------------------------------------

def test_position_from_range_is_equation_5():
    r_s = np.array([6000.0, -1000.0, 2000.0])
    u = np.array([0.6, 0.0, 0.8])
    assert np.array_equal(position_from_range(r_s, u, 0.0), r_s)
    assert np.allclose(position_from_range(r_s, u, 100.0), [6060.0, -1000.0, 2080.0])


def test_tracklet_order_does_not_matter():
    r_s1, u1 = np.array([6378.0, 0, 0]), np.array([1.0, 0, 0])
    r_s2, u2 = np.array([0, 6378.0, 0]), np.array([0, 1.0, 0])
    forward = build_orbit(r_s1, u1, 0.0, 35786.0, r_s2, u2, 21541.0, 35786.0, 0.0)
    reverse = build_orbit(r_s2, u2, 21541.0, 35786.0, r_s1, u1, 0.0, 35786.0, 0.0)

    assert forward[2] == reverse[2] == 21541.0
    assert np.array_equal(forward[0], reverse[0]) and np.array_equal(forward[1], reverse[1])
    assert np.array_equal(forward[3][0].v1, reverse[3][0].v1)


# --------------------------------------------------------------------------
# stage 5: on the generator's data
# --------------------------------------------------------------------------

def test_true_orbit_is_recovered_from_the_tracklets():
    with pipeline(num_objects=5, stages=2) as run:
        worst = 0.0
        count = 0
        for (t1, r1, v1, r_s1, _), (t2, r2, v2, r_s2, _) in _same_object_pairs(run):
            rho_1, rho_2 = np.linalg.norm(r1 - r_s1), np.linalg.norm(r2 - r_s2)
            u_1, u_2 = (r1 - r_s1) / rho_1, (r2 - r_s2) / rho_2
            k = true_k(r1, v1, r2, t2 - t1)

            r_hat_1, r_hat_2, _, candidates = build_orbit(
                r_s1, u_1, t1, rho_1, r_s2, u_2, t2, rho_2, k,
            )
            assert np.allclose(r_hat_1, r1, atol=1e-6) and np.allclose(r_hat_2, r2, atol=1e-6)
            error = min(
                max(np.linalg.norm(sol.v1 - v1), np.linalg.norm(sol.v2 - v2))
                for sol in candidates
            )
            worst = max(worst, error)
            count += 1

    assert count == 50
    assert worst < 1e-8, f"worst velocity error {worst:.2e} km/s"


def test_wrong_k_never_reproduces_the_true_velocity():
    with pipeline(num_objects=3, stages=2) as run:
        for (t1, r1, v1, r_s1, _), (t2, r2, v2, r_s2, _) in _same_object_pairs(run):
            rho_1, rho_2 = np.linalg.norm(r1 - r_s1), np.linalg.norm(r2 - r_s2)
            u_1, u_2 = (r1 - r_s1) / rho_1, (r2 - r_s2) / rho_2
            k = true_k(r1, v1, r2, t2 - t1)
            for wrong in (k + 0.5, k + 1.0, k - 0.5):
                if wrong < 0:
                    continue
                try:
                    *_, candidates = build_orbit(r_s1, u_1, t1, rho_1, r_s2, u_2, t2, rho_2, wrong)
                except LambertNoSolution:
                    continue
                closest = min(np.linalg.norm(sol.v1 - v1) for sol in candidates)
                assert closest > 1e-3, (k, wrong, closest)


# --------------------------------------------------------------------------
# stage 6: simple cases
# --------------------------------------------------------------------------

def test_radial_motion_has_zero_rate():
    rho = np.array([30000.0, 20000.0, 5000.0])
    assert np.allclose(angle_rate_from_state(rho, 0.37 * rho / np.linalg.norm(rho)), 0.0, atol=1e-20)


def test_crossing_motion_has_rate_v_over_r():
    r, v = 36000.0, 3.0
    assert np.allclose(angle_rate_from_state([r, 0, 0], [0, v, 0]), (v / r, 0.0))
    assert np.allclose(angle_rate_from_state([r, 0, 0], [0, 0, v]), (0.0, v / r))
    # At RA = 90 deg, moving towards -x increases RA.
    assert np.allclose(angle_rate_from_state([0, r, 0], [-v, 0, 0]), (v / r, 0.0))


def test_rates_match_finite_differences():
    rng = np.random.default_rng(3)
    for _ in range(100):
        rho = rng.normal(size=3) * 30000.0
        rho_dot = rng.normal(size=3) * 3.0
        h = 1e-2
        a_plus, d_plus = _angles(rho + h * rho_dot)
        a_minus, d_minus = _angles(rho - h * rho_dot)
        numeric = (
            ((a_plus - a_minus + np.pi) % (2 * np.pi) - np.pi) / (2 * h),
            (d_plus - d_minus) / (2 * h),
        )
        assert np.allclose(angle_rate_from_state(rho, rho_dot), numeric, rtol=1e-6, atol=1e-14)


def test_scaling_position_and_velocity_together_keeps_the_rates():
    rho, rho_dot = np.array([20000.0, -30000.0, 4000.0]), np.array([1.0, 2.0, -0.5])
    base = np.array(angle_rate_from_state(rho, rho_dot))
    assert np.allclose(angle_rate_from_state(3 * rho, 3 * rho_dot), base, rtol=1e-12)
    assert np.allclose(angle_rate_from_state(rho, 3 * rho_dot), 3 * base, rtol=1e-12)


def test_site_velocity_is_subtracted():
    """Eq. 17 uses the topocentric velocity: a site moving with the object sees no motion."""
    r, v = np.array([42164.0, 0, 0]), np.array([0, 3.07, 0])
    r_s, v_s = np.array([6378.0, 0, 0]), v.copy()
    assert np.allclose(modeled_angular_rates(r, v, r_s, v_s, r, v, r_s, v_s), 0.0)


# --------------------------------------------------------------------------
# stage 6: edge cases
# --------------------------------------------------------------------------

def test_rate_is_continuous_across_ra_two_pi():
    v = np.array([0.0, 3.0, 0.1])
    below = angle_rate_from_state([36000.0, -1e-3, 100.0], v)
    above = angle_rate_from_state([36000.0, +1e-3, 100.0], v)
    assert np.allclose(below, above, rtol=1e-9)


def test_ra_rate_is_undefined_at_the_pole():
    """s^2 = x^2 + y^2 = 0: alpha_dot divides by zero. The generator's near-GEO
    lines of sight never get close, but the formula has no guard."""
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_dot, delta_dot = angle_rate_from_state([0.0, 0.0, 36000.0], [1.0, 0.0, 0.0])
    assert not (np.isfinite(alpha_dot) and np.isfinite(delta_dot))


# --------------------------------------------------------------------------
# stage 6: on the generator's data
# --------------------------------------------------------------------------

def test_modelled_rates_equal_fitted_clean_rates():
    """Stage 6 at the true state reproduces what stage 2 measured."""
    with pipeline(num_objects=4, stages=2) as run:
        clean = run.attributables("clean")
        worst = without_site = 0.0
        for n, (obj, trk) in enumerate(zip(clean["object_id"], clean["tracklet_id"])):
            _, r, v, r_s, v_s = run.truth_at_epoch(obj, trk)
            measured = clean["attributable"][n][[1, 3]]
            modelled = modeled_angular_rates(r, v, r_s, v_s, r, v, r_s, v_s)[:2]
            ignoring = modeled_angular_rates(r, v, r_s, 0 * v_s, r, v, r_s, 0 * v_s)[:2]
            worst = max(worst, np.max(np.abs(modelled - measured)))
            without_site = max(without_site, np.max(np.abs(ignoring - measured)))

    assert worst < 1e-9, f"{worst / ARCSEC:.2e} arcsec/s"
    # The check has power: dropping the site velocity misses by arcsec/s.
    assert without_site > 1.0 * ARCSEC, f"{without_site / ARCSEC:.2e} arcsec/s"


def test_two_body_rates_are_consistent_over_a_tracklet():
    """The analytic rate integrates to the observed angle change over 80 s."""
    r = np.array([42164.0, 1000.0, 2000.0])
    v = np.array([-0.2, 3.07, 0.3])
    r_s, v_s = np.array([6378.0, 0, 0]), np.array([0, 0.465, 0])

    r_end, _ = two_body(MU, 1.0, r, v)
    a0, d0 = _angles(r - r_s)
    a1, d1 = _angles(r_end - (r_s + v_s))
    alpha_dot, delta_dot = angle_rate_from_state(r - r_s, v - v_s)
    assert np.isclose(a1 - a0, alpha_dot, rtol=1e-4)
    assert np.isclose(d1 - d0, delta_dot, rtol=1e-3)


TESTS = [
    test_position_from_range_is_equation_5,
    test_tracklet_order_does_not_matter,
    test_true_orbit_is_recovered_from_the_tracklets,
    test_wrong_k_never_reproduces_the_true_velocity,
    test_radial_motion_has_zero_rate,
    test_crossing_motion_has_rate_v_over_r,
    test_rates_match_finite_differences,
    test_scaling_position_and_velocity_together_keeps_the_rates,
    test_site_velocity_is_subtracted,
    test_rate_is_continuous_across_ra_two_pi,
    test_ra_rate_is_undefined_at_the_pole,
    test_modelled_rates_equal_fitted_clean_rates,
    test_two_body_rates_are_consistent_over_a_tracklet,
]


if __name__ == "__main__":
    raise SystemExit(run_tests(TESTS))
