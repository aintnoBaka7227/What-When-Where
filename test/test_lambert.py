"""Tests for src/lab/model/lambert.py.

    uv run python test/test_lambert.py

Expected values never come from the code under test.

  1. External reference   Curtis Example 5.2.
  2. Round trip           Recover a known orbit's velocities from its endpoints.
  3. Endpoint closure     Every candidate flies r1 -> r2 in tof (two_body). The
                          only check on the second multi-revolution candidate.
  4. Independent physics  The same with scipy integrating Newton's law.
  5. Structure            Candidates distinct, self-consistent, swept as k says.
  6. Failure paths        Singular and infeasible geometry is refused.
  7. The wrapper          What lambert.py adds over hapsira.core.iod.izzo:
                          untouched velocities, branch chosen by geometry not
                          by hapsira's `prograde` flag, one exception type.
"""


import numpy as np
from hapsira.core.iod import izzo as hapsira_izzo
from scipy.integrate import solve_ivp

from lab.model.lambert import (
    lambert_boundary_value,
    LambertNoSolution,
    k_to_revolutions,
)
from lab.data.gcrs_data_generation import two_body, mu as MU


# Closure tolerances: stable across platforms, but a 1 mm/s error still fails.
POSITION_TOL_KM = 1.0e-3
VELOCITY_TOL_KM_S = 1.0e-5

# Velocity agreement required when recovering a known orbit.
TRUTH_TOL_KM_S = 1.0e-5

# (k, time of flight in periods). Up to 4.5, well past what the pipeline uses.
K_CASES = [
    (0.0, 0.35),   # zero revolutions, short way
    (0.5, 0.65),   # zero revolutions, long way
    (1.0, 1.35),   # one revolution, short way
    (1.5, 1.65),   # one revolution, long way
    (3.5, 3.65),   # three revolutions, long way
    (4.0, 4.35),   # four revolutions, short way
    (4.5, 4.65),   # four revolutions, long way
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_orbit(a_periapsis=42164.0, eccentricity=0.0, inclination_deg=45.0):
    """A known orbit with r1 at periapsis.

    Circular speed times sqrt(1 + e), perpendicular to r, gives periapsis r and
    eccentricity e. e = 0 puts revolutions on whole periods.
    """
    inclination = np.deg2rad(inclination_deg)

    r1 = np.array([a_periapsis, 0.0, 0.0])
    speed = np.sqrt(MU / a_periapsis) * np.sqrt(1.0 + eccentricity)

    v1 = np.array([
        0.0,
        speed * np.cos(inclination),
        speed * np.sin(inclination),
    ])

    semi_major_axis = a_periapsis / (1.0 - eccentricity)
    period = 2.0 * np.pi * np.sqrt(semi_major_axis**3 / MU)

    return r1, v1, period


def _two_body_ode(t, state):
    """Newton's law of gravitation as a first-order system."""
    position = state[:3]
    velocity = state[3:]
    return np.concatenate([
        velocity,
        -MU * position / np.linalg.norm(position) ** 3,
    ])


def integrate_state(r1, v1, tof):
    """Propagate (r1, v1) by tof with scipy -- independent of two_body."""
    solution = solve_ivp(
        _two_body_ode,
        [0.0, tof],
        np.concatenate([r1, v1]),
        method="DOP853",
        rtol=1e-12,
        atol=1e-9,
    )
    assert solution.success, solution.message
    return solution.y[:3, -1], solution.y[3:, -1]


def orbital_elements(r, v):
    """(a, e, |h|) from a Cartesian state, to check two endpoints agree."""
    r_norm = np.linalg.norm(r)
    a = 1.0 / (2.0 / r_norm - v @ v / MU)
    h = np.cross(r, v)
    e = np.linalg.norm(np.cross(v, h) / MU - r / r_norm)
    return a, e, np.linalg.norm(h)


def swept_angle(r1, r2, v1):
    """Swept angle r1 -> r2 in the orbit's own direction.

    Recomputed here rather than calling lambert._is_long_way, which the solver
    itself uses to filter candidates.
    """
    h = np.cross(r1, v1)
    denominator = np.linalg.norm(r1) * np.linalg.norm(r2)

    cos_sweep = np.dot(r1, r2) / denominator
    sin_sweep = np.dot(np.cross(r1, r2), h / np.linalg.norm(h)) / denominator

    return np.arctan2(sin_sweep, np.clip(cos_sweep, -1.0, 1.0)) % (2.0 * np.pi)


def expects_long_way(k):
    """Whether k asks for a long-way transfer, derived from k's own arithmetic.

    Does not call k_to_revolutions() -- that is part of what is being tested.
    """
    return bool(np.isclose(k - np.floor(k), 0.5))


# --------------------------------------------------------------------------
# 1. external reference
# --------------------------------------------------------------------------

def test_curtis_example_5_2():
    """Published worked example: the only ground truth external to this repo."""
    mu_curtis = 398600.0
    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])

    # The unpack asserts exactly one candidate for a zero-revolution transfer.
    (sol,) = lambert_boundary_value(mu_curtis, r1, r2, tof=3600.0, k=0.0)

    assert np.allclose(sol.v1, [-5.9925, 1.9254, 3.2456], atol=1e-3), sol.v1
    assert np.allclose(sol.v2, [-3.3125, -4.1966, -0.3853], atol=1e-3), sol.v2


# --------------------------------------------------------------------------
# 2. round-trip
# --------------------------------------------------------------------------

def test_round_trip_recovers_known_orbit():
    """Feeding a known orbit's endpoints back in must recover its velocities."""
    r1, v1_true, period = make_orbit()

    for k, period_multiplier in K_CASES:
        tof = period_multiplier * period
        r2, v2_true = two_body(MU, tof, r1, v1_true)

        candidates = lambert_boundary_value(MU, r1, r2, tof, k)

        assert any(
            np.allclose(sol.v1, v1_true, atol=TRUTH_TOL_KM_S)
            and np.allclose(sol.v2, v2_true, atol=TRUTH_TOL_KM_S)
            for sol in candidates
        ), f"k={k}: no candidate recovered the known orbit"


def test_round_trip_over_eccentric_orbits():
    """Random near-GEO orbits with e up to 0.05; the k cases above are all circular."""
    rng = np.random.default_rng(20260821)
    failures = []

    for _ in range(60):
        r1, v1_true, period = make_orbit(
            a_periapsis=rng.uniform(38000.0, 45000.0),
            eccentricity=rng.uniform(0.0, 0.05),
            inclination_deg=rng.uniform(1.0, 179.0),
        )

        tof = rng.uniform(0.10, 0.45) * period
        r2, v2_true = two_body(MU, tof, r1, v1_true)

        try:
            candidates = lambert_boundary_value(MU, r1, r2, tof, 0.0)
        except LambertNoSolution as exc:
            failures.append(f"no solution: {exc}")
            continue

        best = min(np.linalg.norm(sol.v1 - v1_true) for sol in candidates)
        if best > TRUTH_TOL_KM_S:
            failures.append(f"velocity error {best:.3e} km/s")

    assert not failures, failures[:5]


# --------------------------------------------------------------------------
# 3. endpoint closure -- every candidate
# --------------------------------------------------------------------------

def test_every_candidate_connects_the_endpoints():
    """Each candidate must fly from r1 to r2 in tof.

    The only check on the second multi-revolution candidate: a candidate that
    misses r2 passes every other test here.
    """
    r1, v1_true, period = make_orbit()

    for k, period_multiplier in K_CASES:
        tof = period_multiplier * period
        r2, _ = two_body(MU, tof, r1, v1_true)

        for index, sol in enumerate(lambert_boundary_value(MU, r1, r2, tof, k)):
            v1, v2 = sol.v1, sol.v2
            r_arrival, v_arrival = two_body(MU, tof, r1, v1)

            position_error = np.linalg.norm(r_arrival - r2)
            velocity_error = np.linalg.norm(v_arrival - v2)

            assert position_error < POSITION_TOL_KM, (
                f"k={k}, candidate {index}: misses r2 by {position_error:.6e} km"
            )
            assert velocity_error < VELOCITY_TOL_KM_S, (
                f"k={k}, candidate {index}: v2 off by {velocity_error:.6e} km/s"
            )


# --------------------------------------------------------------------------
# 4. the same closure, against physics rather than our own propagator
# --------------------------------------------------------------------------

def test_every_candidate_closes_under_independent_integration():
    """The closure check with scipy, so a convention error shared with two_body cannot cancel."""
    r1, v1_true, period = make_orbit()

    for k, period_multiplier in K_CASES:
        tof = period_multiplier * period
        r2, _ = two_body(MU, tof, r1, v1_true)

        for index, sol in enumerate(lambert_boundary_value(MU, r1, r2, tof, k)):
            v1, v2 = sol.v1, sol.v2
            r_arrival, v_arrival = integrate_state(r1, v1, tof)

            position_error = np.linalg.norm(r_arrival - r2)
            velocity_error = np.linalg.norm(v_arrival - v2)

            assert position_error < POSITION_TOL_KM, (
                f"k={k}, candidate {index}: misses r2 by {position_error:.6e} km"
            )
            assert velocity_error < VELOCITY_TOL_KM_S, (
                f"k={k}, candidate {index}: v2 off by {velocity_error:.6e} km/s"
            )


# --------------------------------------------------------------------------
# 5. structural properties
# --------------------------------------------------------------------------

def test_candidates_are_self_consistent_orbits():
    """Each candidate must describe ONE orbit: same a, e, h at both ends."""
    r1, v1_true, period = make_orbit()

    for k, period_multiplier in K_CASES:
        tof = period_multiplier * period
        r2, _ = two_body(MU, tof, r1, v1_true)

        for index, sol in enumerate(lambert_boundary_value(MU, r1, r2, tof, k)):
            v1, v2 = sol.v1, sol.v2
            start = orbital_elements(r1, v1)
            end = orbital_elements(r2, v2)

            assert np.allclose(start, end, rtol=1e-8), (
                f"k={k}, candidate {index}: elements differ, {start} vs {end}"
            )


def test_multi_revolution_returns_two_distinct_candidates():
    """Multi-revolution transfers have distinct low-path and high-path solutions."""
    r1, v1_true, period = make_orbit()

    for k, period_multiplier in K_CASES:
        if k < 1.0:
            continue

        tof = period_multiplier * period
        r2, _ = two_body(MU, tof, r1, v1_true)
        candidates = lambert_boundary_value(MU, r1, r2, tof, k)

        assert len(candidates) == 2, (
            f"k={k}: expected 2 candidates, received {len(candidates)}"
        )

        separation = np.linalg.norm(candidates[0].v1 - candidates[1].v1)
        assert separation > 1e-6, (
            f"k={k}: low/high candidates are duplicates"
        )

        assert {c.low_path for c in candidates} == {True, False}, (
            f"k={k}: expected one low-path and one high-path candidate, "
            f"got {[c.low_path for c in candidates]}"
        )


def test_transfer_angle_matches_k():
    """k's fractional part claims short (<180 deg) or long (>180 deg). Verify it."""
    r1, v1_true, period = make_orbit()

    for k, period_multiplier in K_CASES:
        tof = period_multiplier * period
        r2, _ = two_body(MU, tof, r1, v1_true)

        for index, sol in enumerate(lambert_boundary_value(MU, r1, r2, tof, k)):
            sweep = swept_angle(r1, r2, sol.v1)

            assert (sweep > np.pi) == expects_long_way(k), (
                f"k={k}, candidate {index}: sweep {np.rad2deg(sweep):.3f} deg"
            )
            assert sol.long_way == expects_long_way(k), (
                f"k={k}, candidate {index}: long_way flag disagrees with k"
            )
            assert sol.prograde == (np.cross(r1, sol.v1)[2] > 0.0), (
                f"k={k}, candidate {index}: prograde flag disagrees with h_z"
            )


def test_k_decoding():
    """k's integer part is the revolution count, its fractional part the direction."""
    assert k_to_revolutions(0.0) == (0, False)
    assert k_to_revolutions(0.5) == (0, True)
    assert k_to_revolutions(1.0) == (1, False)
    assert k_to_revolutions(1.5) == (1, True)
    assert k_to_revolutions(4.0) == (4, False)
    assert k_to_revolutions(4.5) == (4, True)


def test_non_half_integer_k_is_rejected():
    """k must be a multiple of 0.5.

    Negative k is not tested: k_to_revolutions(-0.5) gives (-1, True), but no
    stage produces a negative k and the solver refuses it.
    """
    for bad_k in [0.25, 1.3, 0.75]:
        try:
            k_to_revolutions(bad_k)
        except ValueError:
            continue
        raise AssertionError(f"k={bad_k} should have been rejected")


# --------------------------------------------------------------------------
# 6. failure paths
# --------------------------------------------------------------------------

def test_exactly_180_degree_transfer_is_refused():
    """Antipodal endpoints leave the plane undefined; an arbitrary answer would go unnoticed."""
    r1, _, period = make_orbit()
    r2 = -r1

    for k in [0.0, 0.5]:
        try:
            lambert_boundary_value(MU, r1, r2, 0.5 * period, k)
        except LambertNoSolution:
            continue
        raise AssertionError(f"k={k}: returned a solution for a 180 deg transfer")


def test_infeasible_multi_revolution_is_refused():
    """Below the minimum time of flight for N revolutions there is no solution."""
    r1, v1_true, period = make_orbit()

    for k, period_multiplier in [(1.0, 0.60), (2.0, 1.60), (3.0, 2.60)]:
        tof = period_multiplier * period
        r2, _ = two_body(MU, tof, r1, v1_true)

        try:
            lambert_boundary_value(MU, r1, r2, tof, k)
        except LambertNoSolution:
            continue
        raise AssertionError(
            f"k={k}: accepted tof={period_multiplier}P, below the {int(k)}-rev minimum"
        )


def test_non_positive_time_of_flight_is_refused():
    """Two tracklets at the same epoch are never associable via Lambert."""
    r1, v1_true, period = make_orbit()
    r2, _ = two_body(MU, 0.35 * period, r1, v1_true)

    for tof in [0.0, -100.0]:
        try:
            lambert_boundary_value(MU, r1, r2, tof, 0.0)
        except LambertNoSolution:
            continue
        raise AssertionError(f"tof={tof} should have been rejected")


def test_near_singular_geometry_is_refused_or_correct():
    """Near 180 deg the solver may refuse, but any answer must close.

    The optimiser can walk into this geometry.
    """
    r1, v1_true, period = make_orbit()

    for sweep_deg in [179.0, 179.9, 179.99, 180.01, 180.1, 181.0]:
        tof = (sweep_deg / 360.0) * period
        r2, _ = two_body(MU, tof, r1, v1_true)
        k = 0.0 if sweep_deg < 180.0 else 0.5

        try:
            candidates = lambert_boundary_value(MU, r1, r2, tof, k)
        except LambertNoSolution:
            continue

        for index, sol in enumerate(candidates):
            r_arrival, _ = integrate_state(r1, sol.v1, tof)
            error = np.linalg.norm(r_arrival - r2)

            assert error < POSITION_TOL_KM, (
                f"sweep={sweep_deg} deg, candidate {index}: "
                f"misses r2 by {error:.6e} km"
            )


# --------------------------------------------------------------------------
# 7. what the wrapper adds over hapsira
# --------------------------------------------------------------------------

def _planar_pair(angle_deg, radius=42164.0):
    """r1 on +x, r2 at angle_deg about +z (slightly out of plane)."""
    angle = np.deg2rad(angle_deg)
    r1 = np.array([radius, 0.0, 0.0])
    r2 = radius * np.array([np.cos(angle), np.sin(angle), 0.1])
    return r1, r2


def test_hapsira_prograde_flag_does_not_mean_short_way():
    """Why lambert.py classifies solutions itself.

    hapsira's `prograde` is relative to the z axis: with r2 60 deg clockwise of
    r1, prograde=True is the 300 deg long way. The wrapper must still return
    the short way for k = 0.
    """
    r1, r2 = _planar_pair(-60.0)
    tof = 0.17 * 86164.0

    v1, _ = hapsira_izzo(MU, r1, r2, tof, 0, True, True, 35, 1e-8)
    assert swept_angle(r1, r2, v1) > np.pi, "hapsira's prograde=True went the short way here"

    (sol,) = lambert_boundary_value(MU, r1, r2, tof, k=0.0)
    assert swept_angle(r1, r2, sol.v1) < np.pi
    assert not sol.prograde, "the short way round is retrograde for this geometry"


def test_short_way_is_returned_in_both_rotation_senses():
    tof = 0.17 * 86164.0
    for angle in (60.0, -60.0):
        r1, r2 = _planar_pair(angle)
        (sol,) = lambert_boundary_value(MU, r1, r2, tof, k=0.0)
        (long_sol,) = lambert_boundary_value(MU, r1, r2, tof, k=0.5)
        assert swept_angle(r1, r2, sol.v1) < np.pi, angle
        assert swept_angle(r1, r2, long_sol.v1) > np.pi, angle
        assert sol.prograde == (angle > 0) and long_sol.prograde == (angle < 0), angle


def test_wrapper_returns_hapsira_velocities_unchanged():
    """Every candidate is bit-identical to one raw hapsira call."""
    rng = np.random.default_rng(7)
    for _ in range(40):
        r1, v1_true, period = make_orbit(
            a_periapsis=rng.uniform(38000.0, 45000.0),
            eccentricity=rng.uniform(0.0, 0.01),
            inclination_deg=rng.uniform(0.0, 20.0),
        )
        tof = rng.uniform(0.1, 1.9) * period
        r2, _ = two_body(MU, tof, r1, v1_true)
        k = np.floor(2.0 * tof / period) / 2.0
        M = int(k)

        raw = []
        for low_path in ((True,) if M == 0 else (True, False)):
            for prograde in (True, False):
                try:
                    raw.append(hapsira_izzo(MU, r1, r2, tof, M, prograde, low_path, 35, 1e-8))
                except (ValueError, RuntimeError):
                    pass

        for sol in lambert_boundary_value(MU, r1, r2, tof, k):
            assert any(
                np.array_equal(sol.v1, v1) and np.array_equal(sol.v2, v2) for v1, v2 in raw
            ), f"k={k}: candidate not among hapsira's raw outputs"


def test_hapsira_assert_becomes_lambert_no_solution():
    """hapsira asserts tof > 0; the pipeline needs LambertNoSolution."""
    r1, r2 = _planar_pair(60.0)
    try:
        hapsira_izzo(MU, r1, r2, 0.0, 0, True, True, 35, 1e-8)
    except AssertionError:
        pass
    else:
        raise AssertionError("hapsira no longer asserts on tof = 0; revisit the wrapper guard")

    try:
        lambert_boundary_value(MU, r1, r2, 0.0, 0.0)
    except LambertNoSolution:
        return
    raise AssertionError("tof = 0 did not raise LambertNoSolution")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

TESTS = [
    test_curtis_example_5_2,
    test_round_trip_recovers_known_orbit,
    test_round_trip_over_eccentric_orbits,
    test_every_candidate_connects_the_endpoints,
    test_every_candidate_closes_under_independent_integration,
    test_candidates_are_self_consistent_orbits,
    test_multi_revolution_returns_two_distinct_candidates,
    test_transfer_angle_matches_k,
    test_k_decoding,
    test_non_half_integer_k_is_rejected,
    test_exactly_180_degree_transfer_is_refused,
    test_infeasible_multi_revolution_is_refused,
    test_non_positive_time_of_flight_is_refused,
    test_near_singular_geometry_is_refused_or_correct,
    test_hapsira_prograde_flag_does_not_mean_short_way,
    test_short_way_is_returned_in_both_rotation_senses,
    test_wrapper_returns_hapsira_velocities_unchanged,
    test_hapsira_assert_becomes_lambert_no_solution,
]


def main():
    passed = 0
    failed = []

    for test in TESTS:
        name = test.__name__
        try:
            test()
        except AssertionError as exc:
            failed.append(name)
            print(f"  FAIL  {name}\n          {exc}")
        except Exception as exc:
            failed.append(name)
            print(f"  ERROR {name}\n          {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  pass  {name}")

    print()
    print(f"{passed}/{len(TESTS)} passed")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
