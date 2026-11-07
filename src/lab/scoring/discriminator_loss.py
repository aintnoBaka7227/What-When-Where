"""Stage 7: the discriminator loss L(p, k), Siminski et al. Eqs. (16)-(19).

    z_dot   = (alpha_dot_1, delta_dot_1, alpha_dot_2, delta_dot_2)  Eq. (16)
    L(p, k) = (z_dot - z_hat_dot)^T (C_z_dot + C_z_hat_dot)^-1
              (z_dot - z_hat_dot)                                   Eq. (18)
    C_z_hat_dot = J C_z J^T,   J = d z_hat_dot / dz                 Eq. (19)

Two deliberate departures from the printed equations:

* Eq. (18) is printed without the inverse. The paper's own derivation needs
  it: Eq. (21) is a Gaussian likelihood whose Eq. (13) exponent is inverted,
  the text calls Eq. (18) that exponent, and the chi-squared claim holds only
  for a quadratic form in the inverse.
* Eq. (19) is printed J^T C_z J. Propagation through J = dy/dx is J C_z J^T.

z (Eq. 14) stays at the fitted values; only p = (rho_1, rho_2) and k vary.
"""


import numpy as np
from astropy.constants import GM_earth

from lab.data.dataset import NOISY_VARIANTS
from lab.hypothesis.range_bounds import calculate_u_i
from lab.model.build_orbit import build_orbit
from lab.model.model_rates import modeled_angular_rates

# km^3/s^2, as in the generator.
MU = GM_earth.value / 1_000_000_000.0

# Positions in [alpha, alpha_dot, delta, delta_dot].
_ANGLE_INDICES = [0, 2]
_RATE_INDICES = [1, 3]

# The unweighted loss is in arcsec^2/s^2. In rad^2/s^2 near-GEO residuals are
# ~1e-10, below L-BFGS-B's default ftol (~2.2e-9), and the search stops on its
# first step. The weighted loss is dimensionless and needs no scaling.
_RAD_TO_ARCSEC = 180.0 * 3600.0 / np.pi


def measured_z(attributable_1, attributable_2):
    """Eq. (14): z = (alpha_1, delta_1, alpha_2, delta_2)."""
    return np.array([
        attributable_1[0], attributable_1[2],
        attributable_2[0], attributable_2[2],
    ])


def measured_z_dot(attributable_1, attributable_2):
    """Eq. (16): z_dot = (alpha_dot_1, delta_dot_1, alpha_dot_2, delta_dot_2)."""
    return np.array([
        attributable_1[1], attributable_1[3],
        attributable_2[1], attributable_2[3],
    ])


def _block_diag_2x2(a, b):
    """Two 2x2 blocks on the diagonal; tracklets are independent."""
    c = np.zeros((4, 4))
    c[0:2, 0:2] = a
    c[2:4, 2:4] = b
    return c


def measured_z_covariance(ca_1, ca_2):
    """C_z: each tracklet's [alpha, delta] block of Ca."""
    return _block_diag_2x2(
        np.asarray(ca_1)[np.ix_(_ANGLE_INDICES, _ANGLE_INDICES)],
        np.asarray(ca_2)[np.ix_(_ANGLE_INDICES, _ANGLE_INDICES)],
    )


def measured_z_dot_covariance(ca_1, ca_2):
    """C_z_dot: each tracklet's [alpha_dot, delta_dot] block of Ca."""
    return _block_diag_2x2(
        np.asarray(ca_1)[np.ix_(_RATE_INDICES, _RATE_INDICES)],
        np.asarray(ca_2)[np.ix_(_RATE_INDICES, _RATE_INDICES)],
    )


def z_hat_dot_candidates_from_z(
    z, rho_1, rho_2, k, r_s1, v_s1, t1, r_s2, v_s2, t2, mu=MU,
):
    """z_hat_dot for every Lambert candidate, rebuilt from the angles z.

    Takes z rather than u because the Eq. (19) Jacobian perturbs z, so the
    whole z -> r_hat -> Lambert -> rates chain must be redone.
    """
    alpha_1, delta_1, alpha_2, delta_2 = z

    u_1 = calculate_u_i(alpha_1, delta_1)
    u_2 = calculate_u_i(alpha_2, delta_2)

    r_hat_1, r_hat_2, _delta_t, candidates = build_orbit(
        r_s1, u_1, t1, rho_1, r_s2, u_2, t2, rho_2, k, mu=mu,
    )

    return [
        modeled_angular_rates(
            r_hat_1, sol.v1, r_s1, v_s1, r_hat_2, sol.v2, r_s2, v_s2,
        )
        for sol in candidates
    ]


def z_hat_dot_jacobian(
    z, rho_1, rho_2, k, r_s1, v_s1, t1, r_s2, v_s2, t2, mu=MU, eps=1e-6,
):
    """Eq. (19)'s d(z_hat_dot)/dz by central differences, p and k fixed.

    One 4x4 Jacobian per candidate, in candidate order. Raises RuntimeError
    if a perturbation changes the candidate count (near an exact 180 deg
    transfer), rather than pairing the wrong candidates.
    """
    baseline = z_hat_dot_candidates_from_z(
        z, rho_1, rho_2, k, r_s1, v_s1, t1, r_s2, v_s2, t2, mu,
    )
    n_candidates = len(baseline)
    jacobians = [np.zeros((4, 4)) for _ in range(n_candidates)]

    for m in range(4):
        z_plus = np.array(z, dtype=np.float64)
        z_plus[m] += eps
        z_minus = np.array(z, dtype=np.float64)
        z_minus[m] -= eps

        candidates_plus = z_hat_dot_candidates_from_z(
            z_plus, rho_1, rho_2, k, r_s1, v_s1, t1, r_s2, v_s2, t2, mu,
        )
        candidates_minus = z_hat_dot_candidates_from_z(
            z_minus, rho_1, rho_2, k, r_s1, v_s1, t1, r_s2, v_s2, t2, mu,
        )

        if len(candidates_plus) != n_candidates or len(candidates_minus) != n_candidates:
            raise RuntimeError(
                f"Candidate count changed while perturbing z[{m}] for the "
                "Eq. (19) Jacobian -- likely near a feasibility boundary "
                "(e.g. an exact 180-degree transfer angle)."
            )

        for c in range(n_candidates):
            jacobians[c][:, m] = (candidates_plus[c] - candidates_minus[c]) / (2 * eps)

    return jacobians


def rate_residuals(
    rho_1, rho_2, k,
    r_s1, v_s1, t1, attributable_1,
    r_s2, v_s2, t2, attributable_2,
    mu=MU,
):
    """(z, z_dot, [z_dot - z_hat_dot per candidate]), shared by both losses."""
    z = measured_z(attributable_1, attributable_2)
    z_dot = measured_z_dot(attributable_1, attributable_2)

    z_hat_dot_candidates = z_hat_dot_candidates_from_z(
        z, rho_1, rho_2, k, r_s1, v_s1, t1, r_s2, v_s2, t2, mu,
    )

    residuals = [z_dot - z_hat_dot for z_hat_dot in z_hat_dot_candidates]

    return z, z_dot, residuals


def discriminator_loss(
    rho_1, rho_2, k,
    r_s1, v_s1, t1, attributable_1, ca_1,
    r_s2, v_s2, t2, attributable_2, ca_2,
    mu=MU, eps=1e-6,
):
    """Eq. (18), weighted, for the noisy chains. One loss per Lambert candidate."""
    z, z_dot, residuals = rate_residuals(
        rho_1, rho_2, k,
        r_s1, v_s1, t1, attributable_1,
        r_s2, v_s2, t2, attributable_2,
        mu,
    )

    c_z = measured_z_covariance(ca_1, ca_2)
    c_z_dot = measured_z_dot_covariance(ca_1, ca_2)

    jacobians = z_hat_dot_jacobian(
        z, rho_1, rho_2, k, r_s1, v_s1, t1, r_s2, v_s2, t2, mu, eps,
    )

    losses = []
    for residual, jacobian in zip(residuals, jacobians):
        c_z_hat_dot = jacobian @ c_z @ jacobian.T
        c_total = c_z_dot + c_z_hat_dot

        loss = residual @ np.linalg.solve(c_total,residual)
        losses.append(float(loss))

    return losses


def discriminator_loss_unweighted(
    rho_1, rho_2, k,
    r_s1, v_s1, t1, attributable_1,
    r_s2, v_s2, t2, attributable_2,
    mu=MU,
):
    """|z_dot - z_hat_dot|^2 in arcsec^2/s^2, for the clean chain.

    Clean has zero Ca, so the weighted form would invert a singular matrix.
    No Jacobian is needed, so this costs one Lambert solve instead of nine.
    Not comparable with the weighted loss. One loss per candidate.
    """
    _, _, residuals = rate_residuals(
        rho_1, rho_2, k,
        r_s1, v_s1, t1, attributable_1,
        r_s2, v_s2, t2, attributable_2,
        mu,
    )

    return [
        float((residual * _RAD_TO_ARCSEC) @ (residual * _RAD_TO_ARCSEC))
        for residual in residuals
    ]


LOSS_FUNCTIONS = {
    "clean": discriminator_loss_unweighted,
    **{variant: discriminator_loss for variant in NOISY_VARIANTS},
}


def evaluate_loss(
    variant,
    rho_1, rho_2, k,
    r_s1, v_s1, t1, attributable_1, ca_1,
    r_s2, v_s2, t2, attributable_2, ca_2,
    mu=MU, eps=1e-6,
):
    """The chain's loss, with one signature. The unweighted loss ignores Ca."""
    if LOSS_FUNCTIONS[variant] is discriminator_loss_unweighted:
        return discriminator_loss_unweighted(
            rho_1, rho_2, k,
            r_s1, v_s1, t1, attributable_1,
            r_s2, v_s2, t2, attributable_2,
            mu=mu,
        )

    return discriminator_loss(
        rho_1, rho_2, k,
        r_s1, v_s1, t1, attributable_1, ca_1,
        r_s2, v_s2, t2, attributable_2, ca_2,
        mu=mu, eps=eps,
    )


# Tests: uv run python test/test_discriminator_loss.py
