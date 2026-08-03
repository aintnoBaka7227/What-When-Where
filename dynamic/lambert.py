"""Boundary-value Lambert solver for the Siminski et al. tracklet-association method.

Wraps hapsira's Izzo (2015) Lambert's-problem solver (hapsira.core.iod.izzo) and
translates the paper's revolutions/path parameter k (Siminski et al., Eqs. 29-30)
into hapsira's (M, prograde, lowpath) arguments.

Siminski et al. define k as the number of full and half orbital revolutions
between two hypothesized position vectors r(rho_1) and r(rho_2):

    k in {k_min, k_min + 0.5, ..., k_max}

If the fractional part of k is 0, the short-way transfer (transfer angle
< 180 deg) is used; if 0.5, the long-way transfer (> 180 deg) is used
(Section 5, and Fig. 6). The integer part of k is the number of full
revolutions completed during the transfer.

We call hapsira.core.iod.izzo directly, not the astropy-Quantity-wrapped
hapsira.iod.izzo.lambert, because this solver will be evaluated many times per
tracklet pair inside the eventual boundary-value optimizer (Section 5.2), and
the paper's whole point is that this optimization is cheap. hapsira.core.iod
is numba-jitted and takes plain float/ndarray km, km/s inputs, matching the
convention already used by simulation/new_data_gen.py's two_body(). This is an
internal (non-`__all__`) hapsira module, so hapsira is version-pinned in
pyproject.toml (hapsira>=0.18.0) rather than tracked loosely.

Important: hapsira's `prograde` flag is referenced to a fixed z-axis (it
canonicalizes the transfer-plane normal cross(r1, r2) to have a non-negative
z-component before applying the flag), not to the transfer angle within the
plane defined by r1 and r2. Whether `prograde=True` corresponds to the
paper's short-way or long-way transfer therefore depends on the sign of
cross(r1, r2)[2], which varies per hypothesis. Rather than rely on a
sign-convention formula, this module solves the candidate (prograde,
low_path) combinations and classifies each resulting solution by directly
recomputing its true swept transfer angle from r1, r2, and the solution's
own orbital-plane normal (h = r1 x v1) -- this is independent of hapsira's
internal conventions.
"""

import numpy as np
from hapsira.core.iod import izzo as _izzo_core


class LambertNoSolution(RuntimeError):
    """No two-body transfer connects r1 and r2 in tof at revolution count k."""


def k_to_revolutions(k):
    """Split the paper's k (Eqs. 29-30) into (M, long_way).

    M is the number of full revolutions (k's integer part). long_way is
    True if k's fractional part is 0.5 (long-way transfer, angle > 180 deg)
    and False if it is 0 (short-way transfer, angle < 180 deg).
    """
    M = int(np.floor(k + 1e-9))
    fractional = k - M

    if np.isclose(fractional, 0.0, atol=1e-6):
        long_way = False
    elif np.isclose(fractional, 0.5, atol=1e-6):
        long_way = True
    else:
        raise ValueError(
            f"k={k} must be a multiple of 0.5, got fractional part {fractional}"
        )

    return M, long_way


def _is_long_way(r1, r2, v1):
    """Classify a converged Lambert solution as short-way or long-way.

    Recomputes the true-anomaly sweep using the solution's own orbital
    plane (h = r1 x v1), so this does not depend on hapsira's internal
    sign conventions for `prograde`.
    """
    h = np.cross(r1, v1)
    r1_norm, r2_norm = np.linalg.norm(r1), np.linalg.norm(r2)

    cos_dnu = np.clip(np.dot(r1, r2) / (r1_norm * r2_norm), -1.0, 1.0)
    sin_dnu = np.dot(np.cross(r1, r2), h / np.linalg.norm(h)) / (r1_norm * r2_norm)
    dnu = np.arctan2(sin_dnu, cos_dnu) % (2 * np.pi)

    return dnu > np.pi


def lambert_boundary_value(mu, r1, r2, tof, k, numiter=35, rtol=1e-8):
    """Solve the boundary-value Lambert problem for one (rho_1, rho_2) hypothesis.

    This is Section 5's core step: given a hypothesized pair of geocentric
    position vectors r1 = r(rho_1) at t1 and r2 = r(rho_2) at t2 (Eq. 15),
    the time of flight tof = t2 - t1, and a candidate revolutions/path
    parameter k, solve Lambert's problem to get the geocentric velocities at
    both epochs. These feed the modeled angular rates z_hat_dot (Eq. 17) used
    by the discriminator loss function (Eq. 18) in the next pipeline stage.

    Parameters
    ----------
    mu : float
        Gravitational parameter (km^3/s^2).
    r1, r2 : array-like, shape (3,)
        Hypothesized geocentric position vectors (km) at t1 and t2.
    tof : float
        Time of flight t2 - t1 (s). Must be positive.
    k : float
        Revolutions/path parameter from the admissible k grid (Eqs. 29-30),
        a multiple of 0.5.
    numiter, rtol : optional
        Passed through to the Householder iteration.

    Returns
    -------
    candidates : list of (v1, v2) ndarray pairs, shape (3,) each
        Geocentric velocity vectors (km/s) at t1 and t2. For M = 0 (k's
        integer part 0) this list always has exactly one entry, since the
        short/long-way transfer angle and the direction of motion together
        fully determine the orbit. For M > 0 it can have two: this is the
        genuine "vacant focus" ambiguity (whether the transfer ellipse's
        empty focus lies below or above the chord line) -- two physically
        distinct orbits can both sweep the same number of revolutions via
        the same short/long-way arc. The paper does not disambiguate
        between them for its geostationary test case, where M is almost
        always 0; picking between multiple candidates is left to the
        eventual discriminator/optimizer stage, which can score each
        against the tracklets' measured angular rates.

    Raises
    ------
    LambertNoSolution
        If no two-body transfer connects r1 and r2 in tof at revolution k
        (e.g. k is outside the feasible range for this geometry, or r1, r2
        are collinear).
    """
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)

    M, long_way = k_to_revolutions(k)
    low_path_options = (True,) if M == 0 else (True, False)

    candidates = []
    for low_path in low_path_options:
        for prograde in (True, False):
            try:
                v1, v2 = _izzo_core(mu, r1, r2, tof, M, prograde, low_path, numiter, rtol)
            except (ValueError, RuntimeError):
                continue
            if _is_long_way(r1, r2, v1) == long_way:
                candidates.append((v1, v2))

    if not candidates:
        raise LambertNoSolution(
            f"No Lambert solution for r1={r1}, r2={r2}, tof={tof}, "
            f"k={k} (M={M}, long_way={long_way})"
        )

    return candidates


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # --- Sanity check against a published example (Curtis, "Orbital
    # Mechanics for Engineering Students", Example 5.2; also used as the
    # docstring example for hapsira.core.iod.vallado). Confirms the wrapper
    # is wired up correctly, independent of the k-mapping logic below. ---
    mu_earth = 398600.0  # km^3/s^2, matches the Curtis example
    r1_curtis = np.array([5000.0, 10000.0, 2100.0])
    r2_curtis = np.array([-14600.0, 2500.0, 7000.0])
    tof_curtis = 3600.0
    expected_v1 = np.array([-5.9925, 1.9254, 3.2456])
    expected_v2 = np.array([-3.3125, -4.1966, -0.3853])

    (v1, v2), = lambert_boundary_value(mu_earth, r1_curtis, r2_curtis, tof_curtis, k=0.0)
    assert np.allclose(v1, expected_v1, atol=1e-3), (v1, expected_v1)
    assert np.allclose(v2, expected_v2, atol=1e-3), (v2, expected_v2)
    print("Curtis example 5.2 check passed:", v1, v2)

    # --- Round-trip check against this project's own two-body propagator,
    # in the near-GEO regime and across the k values the paper actually
    # needs. This exercises the k -> (M, long_way) classification, which
    # the Curtis check above does not. A perfectly circular, inclined orbit
    # is used so that revolution boundaries land exactly on whole periods,
    # while still exercising a non-degenerate (non-equatorial) geometry. ---
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simulation"))
    from new_data_gen import two_body, mu as mu_km  # noqa: E402

    rng = np.random.default_rng(0)

    a = rng.uniform(38000.0, 45000.0)
    period = 2 * np.pi * np.sqrt(a**3 / mu_km)

    incl = np.deg2rad(45.0)
    r1_geo = np.array([a, 0.0, 0.0])
    v_circ = np.sqrt(mu_km / a)
    v1_geo = np.array([0.0, v_circ * np.cos(incl), v_circ * np.sin(incl)])

    # Fractions below 0.5 sweep a true-anomaly change < 180 deg (short way);
    # fractions above 0.5 sweep > 180 deg (long way). The k label must match
    # which one the chosen tof actually produces.
    test_cases = [
        ("short way, 0 rev", 0.35 * period, 0.0),
        ("long way, 0 rev", 0.65 * period, 0.5),
        ("short way, 1 rev", 1.35 * period, 1.0),
        ("long way, 1 rev", 1.65 * period, 1.5),
    ]

    for label, tof, k in test_cases:
        r2_true, v2_true = two_body(mu_km, tof, r1_geo, v1_geo)

        candidates = lambert_boundary_value(mu_km, r1_geo, r2_true, tof, k=k)

        # For M > 0 there can be a second, physically distinct "vacant
        # focus" candidate (see the docstring) -- accept a match against
        # any of them, not just the first.
        ok = any(
            np.allclose(v1_hat, v1_geo, atol=1e-5)
            and np.allclose(v2_hat, v2_true, atol=1e-5)
            for v1_hat, v2_hat in candidates
        )
        print(
            f"{label}: k={k}, tof={tof:.1f}s, "
            f"{len(candidates)} candidate(s) -> {'OK' if ok else 'MISMATCH'}"
        )
        assert ok, f"{label} failed: candidates={candidates} vs v1={v1_geo}"

        # Negative check: an adjacent-but-wrong k must NOT match, proving
        # k actually distinguishes real transfer branches.
        wrong_k = k + 0.5 if k == np.floor(k) else k - 0.5
        try:
            wrong_candidates = lambert_boundary_value(mu_km, r1_geo, r2_true, tof, k=wrong_k)
            mismatched = not any(
                np.allclose(v1_wrong, v1_geo, atol=1e-3) for v1_wrong, _ in wrong_candidates
            )
        except LambertNoSolution:
            mismatched = True
        assert mismatched, f"{label}: wrong k={wrong_k} should not match but did"

    print("All round-trip checks passed.")
