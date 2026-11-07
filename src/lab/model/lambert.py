"""Boundary-value Lambert solver for Siminski et al.'s tracklet association.

Wraps hapsira's Izzo (2015) solver, hapsira.core.iod.izzo, and maps the
paper's revolution parameter k (Eqs. 29-30) onto its (M, prograde, lowpath)
arguments. k's integer part is the number of full revolutions M; its fraction
selects the short way (0, transfer angle < 180 deg) or the long way (0.5,
> 180 deg).

The core function is called directly, not the astropy-Quantity wrapper,
because the optimiser calls it many times per pair: it is numba-jitted and
takes plain km, km/s. It is not in hapsira's __all__, so hapsira is pinned in
pyproject.toml.

hapsira's `prograde` flag is relative to the fixed z axis, not to the
transfer angle, so prograde=True can be either the short or the long way.
This module therefore tries every (prograde, lowpath) combination and keeps
the solutions whose swept angle, recomputed from r1, r2 and the solution's
own h = r1 x v1, matches k. Velocities are returned exactly as hapsira gives
them.
"""

from typing import NamedTuple, Optional

import numpy as np
from hapsira.core.iod import izzo as _izzo_core


class LambertNoSolution(RuntimeError):
    """No two-body transfer connects r1 and r2 in tof at revolution count k."""


class LambertSolution(NamedTuple):
    """One converged transfer and its geometric classification.

    v1, v2 : ndarray, shape (3,)
        Velocities (km/s) at t1 and t2.
    prograde : bool
        h_z > 0, read from the solution rather than hapsira's input flag.
    long_way : bool
        Swept angle above 180 deg. Always matches the requested k.
    low_path : bool or None
        For M > 0, hapsira's low-path (True) or high-path (False) branch: two
        distinct orbits sweeping the same arc. None for M == 0.
    """

    v1: np.ndarray
    v2: np.ndarray
    prograde: bool
    long_way: bool
    low_path: Optional[bool]


def k_to_revolutions(k):
    """Split k into (M, long_way). Raises unless k is a multiple of 0.5."""
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
    """Whether the solution sweeps more than 180 deg, in its own orbit plane."""
    h = np.cross(r1, v1)
    r1_norm, r2_norm = np.linalg.norm(r1), np.linalg.norm(r2)

    cos_dnu = np.clip(np.dot(r1, r2) / (r1_norm * r2_norm), -1.0, 1.0)
    sin_dnu = np.dot(np.cross(r1, r2), h / np.linalg.norm(h)) / (r1_norm * r2_norm)
    dnu = np.arctan2(sin_dnu, cos_dnu) % (2 * np.pi)

    return dnu > np.pi


def _is_prograde(r1, v1):
    """Whether h = r1 x v1 has a positive z-component."""
    return bool(np.cross(r1, v1)[2] > 0.0)


def lambert_boundary_value(mu, r1, r2, tof, k, numiter=35, rtol=1e-8):
    """Solve Lambert's problem for one hypothesis (Section 5).

    Parameters
    ----------
    mu : float
        km^3/s^2.
    r1, r2 : array-like, shape (3,)
        Hypothesised positions (km) at t1 and t2, Eq. (15).
    tof : float
        t2 - t1 (s), positive.
    k : float
        Revolution parameter, a multiple of 0.5.
    numiter, rtol : optional
        Passed to hapsira's Householder iteration.

    Returns
    -------
    list of LambertSolution
        Exactly one for M = 0. Up to two for M > 0, the low-path and
        high-path orbits; the paper does not choose between them, so the
        optimiser scores both.

    Raises
    ------
    LambertNoSolution
        If no transfer exists: k infeasible for this geometry, r1 and r2
        collinear, or tof <= 0.
    """
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)

    if tof <= 0:
        # hapsira only asserts this, which callers cannot catch by type.
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


# Tests: uv run python test/test_lambert.py
