"""Build boundary-value orbit hypotheses from ranged tracklet pairs.

Implements Equation (5) -- the geocentric position implied by a hypothesized
range along a tracklet's fixed line-of-sight direction:

    r(rho) = r_s + rho * u

Given two tracklets' site positions, line-of-sight directions, and epochs
(from hypothesis/range_bounds.py), a range hypothesis p = (rho_1, rho_2),
and a candidate revolutions/path parameter k (from hypothesis/k_bounds.py),
this module builds r_hat_1, r_hat_2 (Eq. 5's r_hat(rho) at t1 and t2) and
delta_t = t2 - t1 (Eq. 29's notation), then hands them to this directory's
lambert.py -- Section 5's "Orbit" step (Fig. 5). Lives in orbit/ because,
like the paper, generating r_hat and r_hat_dot is considered part of orbit
determination; the admissible-region bounds that (rho_1, rho_2, k)
hypotheses are drawn *from* live in hypothesis/ instead.
"""

import sys
from pathlib import Path

import numpy as np
from astropy.constants import GM_earth

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "hypothesis"))

from lambert import lambert_boundary_value, LambertNoSolution  # noqa: E402
from k_bounds import k_grid  # noqa: E402
from k_bounds import PRECISION_OUTPUT_FILES as K_BOUNDS_FILES  # noqa: E402
from range_bounds import VARIANTS  # noqa: E402
from range_bounds import PRECISION_OUTPUT_FILES as RANGE_BOUNDS_FILES  # noqa: E402

# km^3/s^2, matches simulation/new_data_gen.py, hypothesis/k_bounds.py.
MU = GM_earth.value / 1_000_000_000.0


def position_from_range(r_s, u, rho):
    """Equation (5): geocentric position implied by a range hypothesis."""
    return np.asarray(r_s, dtype=np.float64) + rho * np.asarray(u, dtype=np.float64)


def build_orbit(r_s_1, u_1, t1, rho_1, r_s_2, u_2, t2, rho_2, k, mu=MU):
    """Solve the boundary-value orbit for one hypothesis p = (rho_1, rho_2, k).

    Orders the two tracklets by epoch internally, so callers do not need to
    know in advance which one comes first.

    Parameters
    ----------
    r_s_1, r_s_2 : array-like, shape (3,)
        Observing-site geocentric positions (km) at each tracklet's
        attributable epoch.
    u_1, u_2 : array-like, shape (3,)
        Line-of-sight unit vectors (hypothesis/range_bounds.py's
        calculate_u_i) at each tracklet's attributable epoch.
    t1, t2 : float
        Observation epochs (s), matching Eq. (14)'s z = (alpha_1, delta_1,
        alpha_2, delta_2) and Eq. (29)'s delta_t = t2 - t1.
    rho_1, rho_2 : float
        Hypothesized topocentric ranges (km); together, the paper's
        hypothesis variable p = (rho_1, rho_2).
    k : float
        Revolutions/path parameter from the admissible k grid
        (hypothesis/k_bounds.py).

    Returns
    -------
    r_hat_1, r_hat_2 : ndarray, shape (3,)
        Hypothesized geocentric positions (km), Eq. (5)'s r_hat(rho) at t1
        and t2, returned in the order (earlier epoch, later epoch).
    delta_t : float
        t2 - t1 (s); always positive on return.
    candidates : list of (r_hat_dot_1, r_hat_dot_2) ndarray pairs
        Boundary-value Lambert solutions -- the geocentric velocities at
        t1, t2 for each candidate orbit; see this directory's
        lambert.lambert_boundary_value for why this can have more than one
        entry.

    Raises
    ------
    lambert.LambertNoSolution
        If no two-body transfer connects r_hat_1 and r_hat_2 in delta_t at
        revolution k.
    """
    if t1 > t2:
        r_s_1, u_1, t1, rho_1, r_s_2, u_2, t2, rho_2 = (
            r_s_2, u_2, t2, rho_2, r_s_1, u_1, t1, rho_1,
        )

    r_hat_1 = position_from_range(r_s_1, u_1, rho_1)
    r_hat_2 = position_from_range(r_s_2, u_2, rho_2)
    delta_t = t2 - t1

    candidates = lambert_boundary_value(mu, r_hat_1, r_hat_2, delta_t, k)

    return r_hat_1, r_hat_2, delta_t, candidates


def _lookup_tracklet(object_ids, tracklet_ids, object_id, tracklet_id):
    """Find the row index for one (object_id, tracklet_id) in range_bounds arrays."""
    matches = np.flatnonzero((object_ids == object_id) & (tracklet_ids == tracklet_id))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one match for object {object_id} "
            f"tracklet {tracklet_id}, found {len(matches)}"
        )
    return int(matches[0])


def demo_variant(variant):
    """Demonstrate the range_bounds -> k_bounds -> hypothesis -> Lambert chain.

    For each tracklet pair and each k in its admissible grid, this uses the
    midpoint of each tracklet's [rho_min, rho_max] range bounds as a quick
    illustrative (rho_1, rho_2) guess -- NOT the circular-orbit initial
    guess of Eq. (31), which belongs to the optimizer stage and is not
    implemented yet. This only confirms the pieces built so far connect
    correctly end to end; it is not itself an association test.

    Parameters
    ----------
    variant : str
        Which dataset chain to demonstrate, "clean" or "noisy".
    """
    with np.load(RANGE_BOUNDS_FILES[variant]) as archive:
        rb_object_ids = archive["object_id"]
        rb_tracklet_ids = archive["tracklet_id"]
        t0_values = archive["t0"]
        r_s_i = archive["r_s_i_km"]
        u_i = archive["u_i"]
        rho_min_i = archive["rho_min_i_km"]
        rho_max_i = archive["rho_max_i_km"]

    with np.load(K_BOUNDS_FILES[variant]) as archive:
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
                r_hat_1, r_hat_2, delta_t, candidates = build_orbit(
                    r_s_i[i], u_i[i], t0_values[i], rho_1_guess,
                    r_s_i[j], u_i[j], t0_values[j], rho_2_guess,
                    k,
                )
                print(f"  k={k}: {len(candidates)} candidate(s), delta_t={delta_t:.1f}s")
            except LambertNoSolution as exc:
                print(f"  k={k}: no solution ({exc})")


def main():
    """Run the demo for the clean and the noisy chain."""
    for variant in VARIANTS:
        if not (RANGE_BOUNDS_FILES[variant].exists() and K_BOUNDS_FILES[variant].exists()):
            print(f"Skipping {variant}: bounds files missing")
            continue

        print(f"--- {variant} ---")
        demo_variant(variant)
        print()


if __name__ == "__main__":
    main()
