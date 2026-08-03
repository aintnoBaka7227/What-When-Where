"""Build boundary-value orbit hypotheses from ranged tracklet pairs.

Implements Equation (5) -- the geocentric position implied by a hypothesized
range along a tracklet's fixed line-of-sight direction:

    r(rho) = r_s + rho * u

Given two tracklets' site positions, line-of-sight directions, and epochs
(from orbit/range_bounds.py), a range hypothesis (rho_1, rho_2), and a
candidate revolutions/path parameter k (from orbit/k_bounds.py), this module
builds r1, r2, and the time of flight, then hands them to
dynamic/lambert.py's boundary-value Lambert solver -- Section 5's "Orbit"
step (Fig. 5).
"""

import sys
from pathlib import Path

import numpy as np
from astropy.constants import GM_earth

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "dynamic"))
sys.path.insert(0, str(_PROJECT_ROOT / "orbit"))

from lambert import lambert_boundary_value, LambertNoSolution  # noqa: E402
from k_bounds import k_grid  # noqa: E402

RANGE_BOUNDS_FILE = _PROJECT_ROOT / "orbit" / "range_bounds.npz"
K_BOUNDS_FILE = _PROJECT_ROOT / "orbit" / "k_bounds.npz"

# km^3/s^2, matches simulation/new_data_gen.py, orbit/k_bounds.py.
MU = GM_earth.value / 1_000_000_000.0


def position_from_range(r_s, u, rho):
    """Equation (5): geocentric position implied by a range hypothesis."""
    return np.asarray(r_s, dtype=np.float64) + rho * np.asarray(u, dtype=np.float64)


def build_hypothesis(r_s_1, u_1, t0_1, rho_1, r_s_2, u_2, t0_2, rho_2, k, mu=MU):
    """Solve the boundary-value orbit for one (rho_1, rho_2, k) hypothesis.

    Orders the two tracklets by epoch internally, so callers do not need to
    know in advance which one comes first.

    Parameters
    ----------
    r_s_1, r_s_2 : array-like, shape (3,)
        Observing-site geocentric positions (km) at each tracklet's
        attributable epoch.
    u_1, u_2 : array-like, shape (3,)
        Line-of-sight unit vectors (orbit/range_bounds.py's calculate_u_i)
        at each tracklet's attributable epoch.
    t0_1, t0_2 : float
        Attributable reference epochs (s).
    rho_1, rho_2 : float
        Hypothesized topocentric ranges (km).
    k : float
        Revolutions/path parameter from the admissible k grid
        (orbit/k_bounds.py).

    Returns
    -------
    r1, r2 : ndarray, shape (3,)
        Hypothesized geocentric positions (km), Eq. (5), returned in the
        order (earlier epoch, later epoch).
    tof : float
        Time of flight between them (s); always positive.
    candidates : list of (v1, v2) ndarray pairs
        Boundary-value Lambert solutions; see
        dynamic.lambert.lambert_boundary_value for why this can have more
        than one entry.

    Raises
    ------
    dynamic.lambert.LambertNoSolution
        If no two-body transfer connects r1 and r2 in tof at revolution k.
    """
    if t0_1 > t0_2:
        r_s_1, u_1, t0_1, rho_1, r_s_2, u_2, t0_2, rho_2 = (
            r_s_2, u_2, t0_2, rho_2, r_s_1, u_1, t0_1, rho_1,
        )

    r1 = position_from_range(r_s_1, u_1, rho_1)
    r2 = position_from_range(r_s_2, u_2, rho_2)
    tof = t0_2 - t0_1

    candidates = lambert_boundary_value(mu, r1, r2, tof, k)

    return r1, r2, tof, candidates


def _lookup_tracklet(object_ids, tracklet_ids, object_id, tracklet_id):
    """Find the row index for one (object_id, tracklet_id) in range_bounds arrays."""
    matches = np.flatnonzero((object_ids == object_id) & (tracklet_ids == tracklet_id))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one match for object {object_id} "
            f"tracklet {tracklet_id}, found {len(matches)}"
        )
    return int(matches[0])


def main():
    """Demonstrate the range_bounds -> k_bounds -> hypothesis -> Lambert chain.

    For each tracklet pair and each k in its admissible grid, this uses the
    midpoint of each tracklet's [rho_min, rho_max] range bounds as a quick
    illustrative (rho_1, rho_2) guess -- NOT the circular-orbit initial
    guess of Eq. (31), which belongs to the optimizer stage and is not
    implemented yet. This only confirms the pieces built so far connect
    correctly end to end; it is not itself an association test.
    """
    with np.load(RANGE_BOUNDS_FILE) as archive:
        rb_object_ids = archive["object_id"]
        rb_tracklet_ids = archive["tracklet_id"]
        t0_values = archive["t0"]
        r_s_i = archive["r_s_i_km"]
        u_i = archive["u_i"]
        rho_min_i = archive["rho_min_i_km"]
        rho_max_i = archive["rho_max_i_km"]

    with np.load(K_BOUNDS_FILE) as archive:
        pair_object_id_1 = archive["object_id_1"]
        pair_tracklet_id_1 = archive["tracklet_id_1"]
        pair_object_id_2 = archive["object_id_2"]
        pair_tracklet_id_2 = archive["tracklet_id_2"]
        pair_k_min = archive["k_min"]
        pair_k_max = archive["k_max"]

    for idx in range(len(pair_object_id_1)):
        i = _lookup_tracklet(
            rb_object_ids, rb_tracklet_ids, pair_object_id_1[idx], pair_tracklet_id_1[idx],
        )
        j = _lookup_tracklet(
            rb_object_ids, rb_tracklet_ids, pair_object_id_2[idx], pair_tracklet_id_2[idx],
        )

        rho_1_guess = 0.5 * (rho_min_i[i] + rho_max_i[i])
        rho_2_guess = 0.5 * (rho_min_i[j] + rho_max_i[j])

        grid = k_grid(pair_k_min[idx], pair_k_max[idx])

        print(
            f"Object {int(rb_object_ids[i])} tracklet {int(rb_tracklet_ids[i])} <-> "
            f"Object {int(rb_object_ids[j])} tracklet {int(rb_tracklet_ids[j])}:"
        )
        for k in grid:
            try:
                r1, r2, tof, candidates = build_hypothesis(
                    r_s_i[i], u_i[i], t0_values[i], rho_1_guess,
                    r_s_i[j], u_i[j], t0_values[j], rho_2_guess,
                    k,
                )
                print(f"  k={k}: {len(candidates)} candidate(s), tof={tof:.1f}s")
            except LambertNoSolution as exc:
                print(f"  k={k}: no solution ({exc})")


if __name__ == "__main__":
    main()
