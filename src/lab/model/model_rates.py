"""Compute the modeled topocentric angular rates z_hat_dot (Eq. 17).

Given a candidate boundary-value orbit -- the geocentric positions
r_hat_1, r_hat_2 (Eq. 5) and the geocentric velocities r_hat_dot_1,
r_hat_dot_2 returned by src/lab/model/lambert.py's Lambert solve -- together
with the observing site's geocentric position and velocity at each
epoch, this computes the modeled topocentric angular rates

    z_hat_dot = (alpha_hat_dot_1, delta_hat_dot_1,
                 alpha_hat_dot_2, delta_hat_dot_2)

This is the quantity src/lab/scoring/discriminator_loss.py (Eq. 18) compares
against the tracklets' measured angular rates z_dot -- the attributable's
alpha_dot/delta_dot, already sitting in
data/generated_attributables_{clean,noisy}.npz. Lives in its own
model/ directory, separate from scoring/, so "compute the model's
predicted rate" and "score it against the measurement" stay two distinct
pieces, matching Fig. 5's separate "Modeled" and "Test" stages.

This module is pure kinematics: it differentiates a direction, and knows
nothing about orbits, files, or the dataset chains. Everything it needs
arrives as arguments.
"""

import numpy as np


def angle_rate_from_state(rho_vec, rho_dot_vec):
    """Differentiate right ascension/declination to get their rates.

    Given a topocentric (site-relative) position rho_vec and velocity
    rho_dot_vec, expressed in the same inertial frame used throughout this
    project (e.g. src/lab/data/new_data_gen.py's gcrs_to_radec, which gives
    the angles themselves but not their rates), returns the implied
    (alpha_dot, delta_dot).

    Derivation: with alpha = atan2(py, px) and delta = asin(pz / |rho|),
    differentiating both with respect to time and simplifying gives

        alpha_dot = (px * vy - py * vx) / s^2
        delta_dot = (vz * s^2 - pz * (px * vx + py * vy)) / (r^2 * s)

    where s^2 = px^2 + py^2 (the equatorial-plane projection) and
    r^2 = s^2 + pz^2 (the full range, squared).

    Parameters
    ----------
    rho_vec : array-like, shape (3,)
        Topocentric position (satellite minus site), any consistent unit.
    rho_dot_vec : array-like, shape (3,)
        Topocentric velocity (satellite minus site), same unit per second.

    Returns
    -------
    alpha_dot, delta_dot : float
        Right ascension and declination rates (radians per second, if the
        inputs are km and km/s).
    """
    px, py, pz = np.asarray(rho_vec, dtype=np.float64)
    vx, vy, vz = np.asarray(rho_dot_vec, dtype=np.float64)

    s_sq = px**2 + py**2
    r_sq = s_sq + pz**2

    alpha_dot = (px * vy - py * vx) / s_sq
    delta_dot = (vz * s_sq - pz * (px * vx + py * vy)) / (r_sq * np.sqrt(s_sq))

    return alpha_dot, delta_dot


def modeled_angular_rates(r1, v1, r_s1, v_s1, r2, v2, r_s2, v_s2):
    """Equation (17): modeled topocentric rates from a candidate orbit.

    Parameters
    ----------
    r1, r2 : array-like, shape (3,)
        Hypothesized geocentric positions (km) at t1, t2 -- Eq. (5)'s
        r_hat_1, r_hat_2 (e.g. from src/lab/model/build_orbit.py).
    v1, v2 : array-like, shape (3,)
        Geocentric velocities (km/s) at t1, t2 -- the Lambert solve's
        r_hat_dot_1, r_hat_dot_2 (src/lab/model/lambert.py).
    r_s1, r_s2 : array-like, shape (3,)
        Observing-site geocentric positions (km) at t1, t2.
    v_s1, v_s2 : array-like, shape (3,)
        Observing-site geocentric velocities (km/s) at t1, t2.

    Returns
    -------
    z_hat_dot : ndarray, shape (4,)
        [alpha_hat_dot_1, delta_hat_dot_1, alpha_hat_dot_2, delta_hat_dot_2],
        matching the paper's z_dot ordering (Eq. 16).
    """
    rho_1 = np.asarray(r1, dtype=np.float64) - np.asarray(r_s1, dtype=np.float64)
    rho_dot_1 = np.asarray(v1, dtype=np.float64) - np.asarray(v_s1, dtype=np.float64)
    alpha_dot_1, delta_dot_1 = angle_rate_from_state(rho_1, rho_dot_1)

    rho_2 = np.asarray(r2, dtype=np.float64) - np.asarray(r_s2, dtype=np.float64)
    rho_dot_2 = np.asarray(v2, dtype=np.float64) - np.asarray(v_s2, dtype=np.float64)
    alpha_dot_2, delta_dot_2 = angle_rate_from_state(rho_2, rho_dot_2)

    return np.array([alpha_dot_1, delta_dot_1, alpha_dot_2, delta_dot_2])