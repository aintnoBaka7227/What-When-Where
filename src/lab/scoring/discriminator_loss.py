"""Discriminator loss L(p, k) for a boundary-value tracklet-association hypothesis.

Eqs. (16)-(19): score a candidate orbit's modeled rates z_hat_dot
(src/lab/model/model_rates.py) against the tracklets' measured z_dot.

    z_dot   = (alpha_dot_1, delta_dot_1, alpha_dot_2, delta_dot_2)  Eq. (16)

    L(p, k) = (z_dot - z_hat_dot)^T (C_z_dot + C_z_hat_dot)^-1
              (z_dot - z_hat_dot)                                   Eq. (18)

    C_z_hat_dot = J C_z J^T,   J = d z_hat_dot / dz                 Eq. (19)

Two deliberate departures from the printed equations:

* Eq. (18) omits the inverse. The paper's own derivation requires it:
  Eq. (21) makes the likelihood a Gaussian whose Eq. (13) exponent is
  explicitly inverted, the text calls Eq. (18) that exponent, and its
  chi-squared claim holds only for a quadratic form in the inverse.

* Eq. (19) is printed J^T C_z J. Standard covariance propagation for
  J = dy/dx is J C_z J^T, which is what the Jacobian built below is.

z (Eq. 14) stays fixed at the fitted values; only p = (rho_1, rho_2) and k
vary. Its uncertainty enters solely through the Eq. (19) Jacobian.
"""


import numpy as np
from astropy.constants import GM_earth

from lab.data.dataset import NOISY_VARIANTS
from lab.hypothesis.range_bounds import calculate_u_i
from lab.model.build_orbit import build_orbit
from lab.model.model_rates import modeled_angular_rates

# km^3/s^2, matches src/lab/data/gcrs_data_generation.py and friends.
MU = GM_earth.value / 1_000_000_000.0

# Attributable ordering, fixed throughout this project: [alpha, alpha_dot, delta, delta_dot].
_ANGLE_INDICES = [0, 2]
_RATE_INDICES = [1, 3]

# The unweighted loss scores in arcsec/s, not rad/s. Near-GEO rates are
# ~1e-5 rad/s, so squared residuals land near 1e-10 -- under L-BFGS-B's
# default ftol (~2.2e-9), which makes it converge on its first step and
# never leave the initial guess. In arcsec^2/s^2 the landscape spans
# ~1e-3 to ~1e2. The weighted loss needs no rescaling: dividing by a
# covariance in matching units leaves it dimensionless.
_RAD_TO_ARCSEC = 180.0 * 3600.0 / np.pi


def measured_z(attributable_1, attributable_2):
    """Equation (14): z = (alpha_1, delta_1, alpha_2, delta_2)."""
    return np.array([
        attributable_1[0], attributable_1[2],
        attributable_2[0], attributable_2[2],
    ])


def measured_z_dot(attributable_1, attributable_2):
    """Equation (16): z_dot = (alpha_dot_1, delta_dot_1, alpha_dot_2, delta_dot_2)."""
    return np.array([
        attributable_1[1], attributable_1[3],
        attributable_2[1], attributable_2[3],
    ])


def _block_diag_2x2(a, b):
    """Stack two 2x2 covariance blocks block-diagonally.

    The two tracklets have independent noise draws, so the cross blocks
    are zero.
    """
    c = np.zeros((4, 4))
    c[0:2, 0:2] = a
    c[2:4, 2:4] = b
    return c


def measured_z_covariance(ca_1, ca_2):
    """Covariance of z (Eq. 14): each tracklet's [alpha, delta] sub-block of Ca."""
    return _block_diag_2x2(
        np.asarray(ca_1)[np.ix_(_ANGLE_INDICES, _ANGLE_INDICES)],
        np.asarray(ca_2)[np.ix_(_ANGLE_INDICES, _ANGLE_INDICES)],
    )


def measured_z_dot_covariance(ca_1, ca_2):
    """Covariance of z_dot (Eq. 16): each tracklet's [alpha_dot, delta_dot] sub-block of Ca."""
    return _block_diag_2x2(
        np.asarray(ca_1)[np.ix_(_RATE_INDICES, _RATE_INDICES)],
        np.asarray(ca_2)[np.ix_(_RATE_INDICES, _RATE_INDICES)],
    )


def z_hat_dot_candidates_from_z(
    z, rho_1, rho_2, k, r_s1, v_s1, t1, r_s2, v_s2, t2, mu=MU,
):
    """Every candidate z_hat_dot for a hypothesis, rebuilt from z.

    Takes z rather than a precomputed line-of-sight u, because z is what
    the Eq. (19) Jacobian perturbs -- so the whole
    r_hat -> Lambert -> modeled-rate chain has to be redone from it.

    Parameters
    ----------
    z : array-like, shape (4,)
        (alpha_1, delta_1, alpha_2, delta_2).
    rho_1, rho_2 : float
        Range hypothesis (km).
    k : float
        Revolutions/path parameter.
    r_s1, r_s2, v_s1, v_s2 : array-like, shape (3,)
        Site position (km) and velocity (km/s) at t1, t2.
    t1, t2 : float
        Observation epochs (s).

    Returns
    -------
    list of ndarray, shape (4,)
        One z_hat_dot per Lambert candidate.
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
    """Equation (19)'s d(z_hat_dot)/dz, by central finite differences.

    p and k are held fixed; only z is perturbed, one component at a time.

    Returns
    -------
    list of ndarray, shape (4, 4)
        One Jacobian per candidate, matched to
        z_hat_dot_candidates_from_z by list position. Assumes perturbing z
        does not change which branches are feasible -- true away from an
        exact 180-degree transfer. Raises RuntimeError if it does, rather
        than silently pairing the wrong candidates.
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
    """The rate mismatch (z_dot - z_hat_dot) both losses are built on.

    Shared so the two variants cannot drift apart: same residual, and they
    differ only in what multiplies it.

    Returns
    -------
    z : ndarray, shape (4,)
        Eq. (14) angles -- the weighted loss reuses them for the Jacobian.
    z_dot : ndarray, shape (4,)
        Eq. (16) measured rates.
    residuals : list of ndarray, shape (4,)
        z_dot - z_hat_dot, one per Lambert candidate.
    """
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
    """Equation (18): loss L(p, k) for hypothesis p = (rho_1, rho_2).

    Used for the noisy chain. See the module docstring for the matrix
    inverse the printed Eq. (18) omits.

    Returns
    -------
    list of float
        One loss per Lambert candidate. The paper does not say what to do
        with more than one; scoring each and letting the caller pick is
        the natural extension.
    """
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
    """Equation (18) unweighted: L(p, k) = |z_dot - z_hat_dot|^2, in arcsec^2/s^2.

    Used for the clean chain, which has no covariance to weight by: its
    measurement uncertainties are zero, so fit_attributes scales every Ca
    to the zero matrix and the weighted form would invert a singular
    matrix.

    Dropping the weighting also drops the Eq. (19) Jacobian, so this costs
    one Lambert solve per call instead of nine.

    The two losses are NOT comparable across chains -- this one carries
    units, the weighted one is dimensionless. Compare clean to clean and
    noisy to noisy. (It coincides with Eq. 18 *as printed*, but for an
    unrelated reason: the paper means the weighted form.)

    Parameters
    ----------
    As discriminator_loss, without ca_1/ca_2 or eps.

    Returns
    -------
    list of float
        One loss per Lambert candidate, same order as discriminator_loss.
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


# Which loss scores each chain: clean has no covariance to weight by, every
# noisy chain uses the paper's actual method.
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
    """Score a hypothesis with whichever loss this chain requires.

    Covariances are passed unconditionally and ignored by the unweighted
    loss, so callers iterating both chains need no special case.
    """
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


# Validation lives in test/test_discriminator_loss.py, which
# checks the Eq. (18) quadratic form against the algebraic identities it
# must satisfy -- covariance scaling, reduction under C = c*I, and an
# independent solve() -- plus the Eq. (19) Jacobian's step-size stability:
#
#     uv run python test/test_discriminator_loss.py
