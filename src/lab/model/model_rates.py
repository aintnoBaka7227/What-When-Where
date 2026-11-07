"""Modelled topocentric angular rates z_hat_dot, Siminski et al. Eq. (17).

    z_hat_dot = (alpha_hat_dot_1, delta_hat_dot_1, alpha_hat_dot_2, delta_hat_dot_2)

from a candidate orbit's positions and Lambert velocities, minus the site's
position and velocity. scoring/discriminator_loss.py compares these with the
measured rates. Pure kinematics: no files, orbits or chains.
"""

import numpy as np


def angle_rate_from_state(rho_vec, rho_dot_vec):
    """(alpha_dot, delta_dot) from a topocentric position and velocity.

    Differentiating alpha = atan2(py, px) and delta = asin(pz / |rho|):

        alpha_dot = (px vy - py vx) / s^2
        delta_dot = (vz s^2 - pz (px vx + py vy)) / (r^2 s)

    with s^2 = px^2 + py^2 and r^2 = s^2 + pz^2. Undefined at the pole
    (s = 0). rad/s for km and km/s.
    """
    px, py, pz = np.asarray(rho_vec, dtype=np.float64)
    vx, vy, vz = np.asarray(rho_dot_vec, dtype=np.float64)

    s_sq = px**2 + py**2
    r_sq = s_sq + pz**2

    alpha_dot = (px * vy - py * vx) / s_sq
    delta_dot = (vz * s_sq - pz * (px * vx + py * vy)) / (r_sq * np.sqrt(s_sq))

    return alpha_dot, delta_dot


def modeled_angular_rates(r1, v1, r_s1, v_s1, r2, v2, r_s2, v_s2):
    """Eq. (17): z_hat_dot, ordered like z_dot (Eq. 16).

    r1, r2, v1, v2 are the candidate's geocentric positions (km) and
    velocities (km/s) at t1, t2; r_s*, v_s* the site's.
    """
    rho_1 = np.asarray(r1, dtype=np.float64) - np.asarray(r_s1, dtype=np.float64)
    rho_dot_1 = np.asarray(v1, dtype=np.float64) - np.asarray(v_s1, dtype=np.float64)
    alpha_dot_1, delta_dot_1 = angle_rate_from_state(rho_1, rho_dot_1)

    rho_2 = np.asarray(r2, dtype=np.float64) - np.asarray(r_s2, dtype=np.float64)
    rho_dot_2 = np.asarray(v2, dtype=np.float64) - np.asarray(v_s2, dtype=np.float64)
    alpha_dot_2, delta_dot_2 = angle_rate_from_state(rho_2, rho_dot_2)

    return np.array([alpha_dot_1, delta_dot_1, alpha_dot_2, delta_dot_2])
