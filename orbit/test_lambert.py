"""Validation suite for orbit/lambert.py.

Run directly:

    uv run python orbit/test_lambert.py

Every test exercises the real `lambert_boundary_value`. Nothing here
reimplements the solver, and -- importantly -- nothing computes its expected
value by calling the code under test. A test that asks the implementation to
grade its own homework cannot fail.

The layers, each catching what the one above cannot:

  1. External reference   Curtis Example 5.2, published answers. The only
                          ground truth from outside this repository. Catches
                          units, mu, and gross convention errors.
  2. Round-trip           Propagate a known orbit with two_body, feed the
                          endpoints back, recover the velocity. Catches
                          k-decoding and short/long-way errors.
  3. Endpoint closure     EVERY candidate must connect r1 to r2 in tof. This
                          is the definition of a Lambert solution, and it is
                          the only check that validates the second
                          multi-revolution candidate -- which the
                          discriminator scores but the round-trip ignores.
  4. Independent physics  The same closure check, integrating Newton's law
                          with scipy instead of two_body. Depends on neither
                          the solver's internals nor this project's
                          propagator.
  5. Structure            Candidates distinct and self-consistent; transfer
                          angle matches what k claims.
  6. Failure paths        Singular geometry, infeasible multi-revolution
                          transfers, and near-singular geometry must be
                          refused rather than silently fudged.

Layers 3 and 4 check every candidate, not the best-matching one. A
multi-revolution call returns two solutions and `optimize_hypothesis` takes
the minimum loss over both, so a silently invalid second candidate could win
and produce a confident wrong association.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT / "simulation"))

from lambert import (  # noqa: E402
    lambert_boundary_value,
    LambertNoSolution,
    k_to_revolutions,
)
from new_data_gen import two_body, mu as MU  # noqa: E402


# Endpoint closure tolerances. two_body and scipy both land well inside
# these for a near-GEO transfer; they are loose enough to be stable across
# platforms and tight enough that a 1 mm/s velocity error (a 45 m endpoint
# miss) still fails.
POSITION_TOL_KM = 1.0e-3
VELOCITY_TOL_KM_S = 1.0e-5

# Velocity agreement required when recovering a known orbit.
TRUTH_TOL_KM_S = 1.0e-5

# The k values exercised throughout, paired with a time of flight expressed
# in orbital periods. The pipeline itself only reaches k = 1.0; going to 4.5
# exercises the multi-revolution branch well past current use.
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
    """Build a known orbit with r1 at periapsis.

    Scaling circular speed by sqrt(1 + e) with the velocity kept
    perpendicular to r makes r a periapsis of an orbit with exactly that
    eccentricity: vis-viva gives a = r / (1 - e), so a(1 - e) = r.

    An inclined orbit keeps the geometry non-degenerate; e = 0 makes
    revolutions land on exact whole periods, which is why the k cases use it.
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
    """True-anomaly sweep from r1 to r2, in the orbit's own direction.

    Deliberately does NOT call lambert._is_long_way(). The solver filters its
    candidates with that function, so using it here would ask the
    implementation to confirm its own answer -- an assertion that cannot
    fail. This recomputes the geometry from the returned velocity instead.
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
    (v1, v2), = lambert_boundary_value(mu_curtis, r1, r2, tof=3600.0, k=0.0)

    assert np.allclose(v1, [-5.9925, 1.9254, 3.2456], atol=1e-3), v1
    assert np.allclose(v2, [-3.3125, -4.1966, -0.3853], atol=1e-3), v2


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
            np.allclose(v1, v1_true, atol=TRUTH_TOL_KM_S)
            and np.allclose(v2, v2_true, atol=TRUTH_TOL_KM_S)
            for v1, v2 in candidates
        ), f"k={k}: no candidate recovered the known orbit"


def test_round_trip_over_eccentric_orbits():
    """Fuzz across the near-GEO population, including eccentric orbits.

    The k cases above all use e = 0 so revolutions land on whole periods.
    Nothing else would exercise eccentric geometry, which is what the
    pipeline's own population (e up to 0.01) actually contains.
    """
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

        best = min(np.linalg.norm(v1 - v1_true) for v1, _ in candidates)
        if best > TRUTH_TOL_KM_S:
            failures.append(f"velocity error {best:.3e} km/s")

    assert not failures, failures[:5]


# --------------------------------------------------------------------------
# 3. endpoint closure -- every candidate
# --------------------------------------------------------------------------

def test_every_candidate_connects_the_endpoints():
    """Each candidate must actually fly from r1 to r2 in tof.

    This is what makes a solution a Lambert solution, and it is the only
    check that validates the SECOND multi-revolution candidate. Without it a
    candidate missing the target by tens of thousands of kilometres passes
    every other assertion here: it is distinct from the first, it sweeps the
    right way, and the first candidate still recovers truth.
    """
    r1, v1_true, period = make_orbit()

    for k, period_multiplier in K_CASES:
        tof = period_multiplier * period
        r2, _ = two_body(MU, tof, r1, v1_true)

        for index, (v1, v2) in enumerate(lambert_boundary_value(MU, r1, r2, tof, k)):
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
    """Repeat the closure check with scipy integrating Newton's law.

    two_body generated r2 and also verifies it above, so a shared convention
    error would cancel out. Integrating the equations of motion directly
    removes that possibility.
    """
    r1, v1_true, period = make_orbit()

    for k, period_multiplier in K_CASES:
        tof = period_multiplier * period
        r2, _ = two_body(MU, tof, r1, v1_true)

        for index, (v1, v2) in enumerate(lambert_boundary_value(MU, r1, r2, tof, k)):
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

        for index, (v1, v2) in enumerate(lambert_boundary_value(MU, r1, r2, tof, k)):
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

        separation = np.linalg.norm(candidates[0][0] - candidates[1][0])
        assert separation > 1e-6, (
            f"k={k}: low/high candidates are duplicates"
        )


def test_transfer_angle_matches_k():
    """k's fractional part claims short (<180 deg) or long (>180 deg). Verify it."""
    r1, v1_true, period = make_orbit()

    for k, period_multiplier in K_CASES:
        tof = period_multiplier * period
        r2, _ = two_body(MU, tof, r1, v1_true)

        for index, (v1, _) in enumerate(lambert_boundary_value(MU, r1, r2, tof, k)):
            sweep = swept_angle(r1, r2, v1)

            assert (sweep > np.pi) == expects_long_way(k), (
                f"k={k}, candidate {index}: sweep {np.rad2deg(sweep):.3f} deg"
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
    """k must be a multiple of 0.5; anything else is a caller error.

    Note this does not test negative k. k_to_revolutions(-0.5) returns
    (-1, True) -- a negative revolution count, which is meaningless -- but
    lambert_boundary_value refuses it downstream, and hypothesis/k_bounds.py
    never produces a negative k. It is an unreachable input-validation gap,
    not a live bug, so asserting on it here would be testing behaviour the
    function does not promise.
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
    """Antipodal endpoints leave the orbital plane undefined.

    Returning an arbitrary plane would be worse than failing, because
    nothing downstream could tell the difference.
    """
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
    """Transfer angles approaching 180 degrees are ill-conditioned, not undefined.

    The solver may legitimately refuse these. What it must not do is return a
    solution that fails its own boundary conditions: the optimizer can walk
    into this geometry, and a quietly wrong answer there would propagate into
    a real association result.
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

        for index, (v1, _) in enumerate(candidates):
            r_arrival, _ = integrate_state(r1, v1, tof)
            error = np.linalg.norm(r_arrival - r2)

            assert error < POSITION_TOL_KM, (
                f"sweep={sweep_deg} deg, candidate {index}: "
                f"misses r2 by {error:.6e} km"
            )


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
        except Exception as exc:  # noqa: BLE001 -- report and continue
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
