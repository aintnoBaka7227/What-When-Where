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
recomputing its true swept transfer angle -- and its direction of motion
(the sign of h_z, h = r1 x v1) -- from r1, r2, and the solution's own
orbital-plane normal. This is independent of hapsira's internal
conventions.

Each returned solution is a LambertSolution carrying, alongside the two
velocity vectors, this classification: prograde/retrograde, long-way/
short-way, and (for multi-revolution transfers) the low-path/high-path
vacant-focus branch.
"""

from typing import NamedTuple, Optional

import numpy as np
from hapsira.core.iod import izzo as _izzo_core


class LambertNoSolution(RuntimeError):
    """No two-body transfer connects r1 and r2 in tof at revolution count k."""


class LambertSolution(NamedTuple):
    """One converged boundary-value transfer, with its geometric classification.

    v1, v2 : ndarray, shape (3,)
        Geocentric velocity vectors (km/s) at t1 and t2.
    prograde : bool
        True if the transfer orbit's motion is prograde (angular momentum
        r1 x v1 has a positive z-component), False if retrograde. Classified
        from the solution itself, not from hapsira's input flag, so it does
        not depend on hapsira's internal sign conventions.
    long_way : bool
        True if the swept transfer angle exceeds 180 degrees (long-way /
        k fractional part 0.5), False for the short way. Recomputed from
        r1, r2 and the solution's own orbital plane; always matches the
        requested k.
    low_path : bool or None
        For a multi-revolution transfer (M > 0), True if this is hapsira's
        low-path branch (transfer ellipse's vacant focus below the chord
        line) and False for the high-path branch -- the two are physically
        distinct orbits sweeping the same arc. None when M == 0, where the
        branches coincide and the transfer is unique.
    """

    v1: np.ndarray
    v2: np.ndarray
    prograde: bool
    long_way: bool
    low_path: Optional[bool]


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


def _is_prograde(r1, v1):
    """Classify a converged Lambert solution as prograde or retrograde.

    Reads the sign of the specific angular momentum's z-component from the
    solution itself, so -- like `_is_long_way` -- it is independent of
    hapsira's internal sign conventions for its `prograde` input flag.
    """
    return bool(np.cross(r1, v1)[2] > 0.0)


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
    candidates : list of LambertSolution
        Each carries the geocentric velocities (v1, v2) at t1 and t2 plus
        the transfer's geometric classification (prograde/retrograde,
        long-way/short-way, low-path/high-path); see LambertSolution. For
        M = 0 (k's integer part 0) this list always has exactly one entry,
        since the short/long-way transfer angle and the direction of
        motion together fully determine the orbit. For M > 0 it can have
        two: this is the genuine "vacant focus" ambiguity (whether the
        transfer ellipse's empty focus lies below or above the chord line)
        -- two physically distinct orbits can both sweep the same number
        of revolutions via the same short/long-way arc. The paper does not
        disambiguate between them for its geostationary test case, where M
        is almost always 0; picking between multiple candidates is left to
        the eventual discriminator/optimizer stage, which can score each
        against the tracklets' measured angular rates.

    Raises
    ------
    LambertNoSolution
        If no two-body transfer connects r1 and r2 in tof at revolution k
        (e.g. k is outside the feasible range for this geometry, r1, r2
        are collinear, or tof <= 0).
    """
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)

    if tof <= 0:
        # hapsira's core solver only guards this with a bare `assert`,
        # which raises an unhelpful, uncatchable-by-type AssertionError.
        # Two tracklets at the same epoch (tof == 0) are never associable
        # via Lambert's problem, which needs two *distinct* epochs.
        raise LambertNoSolution(f"tof={tof} must be positive")

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
                candidates.append(LambertSolution(
                    v1=v1,
                    v2=v2,
                    prograde=_is_prograde(r1, v1),
                    long_way=long_way,
                    low_path=None if M == 0 else low_path,
                ))

    if not candidates:
        raise LambertNoSolution(
            f"No Lambert solution for r1={r1}, r2={r2}, tof={tof}, "
            f"k={k} (M={M}, long_way={long_way})"
        )

    return candidates


# Validation lives in orbit/test_lambert.py, which exercises this module
# against Curtis's published example, round-trips through the project's own
# two-body propagator, independent integration of Newton's law, and the
# degenerate-geometry failure paths:
#
#     uv run python orbit/test_lambert.py
