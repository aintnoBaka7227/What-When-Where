"""Validation suite for discriminator/discriminator_loss.py.

Run directly:

    uv run python discriminator/test_discriminator_loss.py

Self-contained: the two-tracklet geometry is built here from a known orbit,
so nothing depends on the pipeline having been run or on any .npz file
existing.

The loss is the one place the method can be wrong without anything crashing
-- a mis-transposed covariance or a dropped factor still returns plausible
numbers. So most of these are algebraic identities the loss must satisfy
regardless of geometry:

    scaling      Ca -> s*Ca must give L -> L/s
    reduction    C = c*I must give L = |residual|^2 / c
    agreement    an independent solve() must match the inv() in the module
    units        the unweighted loss must be |residual|^2 * 206265^2

None of these depend on knowing the "right" answer for a hypothesis, which
is what makes them usable: a wrong loss fails them, a merely imprecise one
does not.
"""

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT / "orbit"))
sys.path.insert(0, str(_PROJECT_ROOT / "hypothesis"))
sys.path.insert(0, str(_PROJECT_ROOT / "modelled"))
sys.path.insert(0, str(_PROJECT_ROOT / "simulation"))

from discriminator_loss import (  # noqa: E402
    MU,
    _RAD_TO_ARCSEC,
    LOSS_FUNCTIONS,
    measured_z,
    measured_z_dot,
    measured_z_covariance,
    measured_z_dot_covariance,
    rate_residuals,
    z_hat_dot_jacobian,
    discriminator_loss,
    discriminator_loss_unweighted,
    evaluate_loss,
)
from model_rates import angle_rate_from_state  # noqa: E402
from new_data_gen import two_body  # noqa: E402

EARTH_ROTATION_RATE = 7.292115e-5  # rad/s
EARTH_RADIUS_KM = 6378.137


def build_geometry():
    """A feasible two-tracklet hypothesis, built from a known near-GEO orbit.

    Returns everything the loss functions need, plus the true ranges and a
    k that admits a Lambert solution. The hypothesis does not have to be a
    good one -- the identities tested here hold at any feasible point.
    """
    a = 42164.0
    inclination = np.deg2rad(15.0)
    t1, t2 = 0.0, 43200.0

    r_initial = np.array([a, 0.0, 0.0])
    speed = np.sqrt(MU / a)
    v_initial = np.array([
        0.0,
        speed * np.cos(inclination),
        speed * np.sin(inclination),
    ])

    def site(latitude_deg, longitude_deg, t):
        """An Earth-fixed site, with the velocity its rotation implies."""
        latitude = np.deg2rad(latitude_deg)
        longitude = np.deg2rad(longitude_deg) + EARTH_ROTATION_RATE * t

        position = EARTH_RADIUS_KM * np.array([
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ])
        velocity = np.cross([0.0, 0.0, EARTH_ROTATION_RATE], position)

        return position, velocity

    r_1, v_1 = two_body(MU, t1, r_initial, v_initial)
    r_2, v_2 = two_body(MU, t2, r_initial, v_initial)

    r_s1, v_s1 = site(10.0, 5.0, t1)
    r_s2, v_s2 = site(-20.0, 120.0, t2)

    def attributable(r, v, r_s, v_s):
        """[alpha, alpha_dot, delta, delta_dot] from a true state."""
        rho = r - r_s
        rho_dot = v - v_s

        alpha = np.arctan2(rho[1], rho[0]) % (2 * np.pi)
        delta = np.arcsin(rho[2] / np.linalg.norm(rho))
        alpha_dot, delta_dot = angle_rate_from_state(rho, rho_dot)

        return np.array([alpha, alpha_dot, delta, delta_dot])

    attributable_1 = attributable(r_1, v_1, r_s1, v_s1)
    attributable_2 = attributable(r_2, v_2, r_s2, v_s2)

    rho_1 = np.linalg.norm(r_1 - r_s1)
    rho_2 = np.linalg.norm(r_2 - r_s2)

    return dict(
        rho_1=rho_1, rho_2=rho_2, k=0.5,
        r_s1=r_s1, v_s1=v_s1, t1=t1, attributable_1=attributable_1,
        r_s2=r_s2, v_s2=v_s2, t2=t2, attributable_2=attributable_2,
    )


def realistic_covariance(relative_arcsec=0.7, systematic_arcsec=5.0):
    """An attributable covariance shaped like fit_attributes() produces.

    Angle variances carry the systematic term, rate variances do not, and
    the angle-rate off-diagonals are non-zero -- which matters, because a
    diagonal Ca would hide a transpose error.
    """
    sigma_relative = np.deg2rad(relative_arcsec / 3600.0)
    sigma_systematic = np.deg2rad(systematic_arcsec / 3600.0)

    times = np.arange(5) * 30.0
    dt = times - times.mean()
    design = np.column_stack([np.ones_like(dt), dt])
    fit = sigma_relative**2 * np.linalg.inv(design.T @ design)

    block = fit.copy()
    block[0, 0] += sigma_systematic**2

    ca = np.zeros((4, 4))
    ca[np.ix_([0, 1], [0, 1])] = block
    ca[np.ix_([2, 3], [2, 3])] = block
    return ca


def loss_arguments(geometry):
    """Positional arguments shared by both loss functions, minus covariances."""
    return (
        geometry["rho_1"], geometry["rho_2"], geometry["k"],
        geometry["r_s1"], geometry["v_s1"], geometry["t1"], geometry["attributable_1"],
        geometry["r_s2"], geometry["v_s2"], geometry["t2"], geometry["attributable_2"],
    )


def weighted(geometry, ca_1, ca_2):
    g = geometry
    return discriminator_loss(
        g["rho_1"], g["rho_2"], g["k"],
        g["r_s1"], g["v_s1"], g["t1"], g["attributable_1"], ca_1,
        g["r_s2"], g["v_s2"], g["t2"], g["attributable_2"], ca_2,
    )


# --------------------------------------------------------------------------
# ordering: Eqs. (14) and (16)
# --------------------------------------------------------------------------

def test_z_and_z_dot_ordering():
    """Eq. (14) takes the angles, Eq. (16) the rates, interleaved by tracklet."""
    a1 = np.array([1.0, 2.0, 3.0, 4.0])
    a2 = np.array([5.0, 6.0, 7.0, 8.0])

    assert np.array_equal(measured_z(a1, a2), [1.0, 3.0, 5.0, 7.0])
    assert np.array_equal(measured_z_dot(a1, a2), [2.0, 4.0, 6.0, 8.0])


def test_covariance_sub_blocks():
    """C_z takes the [alpha, delta] entries, C_z_dot the [alpha_dot, delta_dot]."""
    ca = np.arange(16, dtype=float).reshape(4, 4)

    c_z = measured_z_covariance(ca, ca)
    c_z_dot = measured_z_dot_covariance(ca, ca)

    # Rows/cols 0 and 2 of Ca are the angles; 1 and 3 are the rates.
    assert np.array_equal(c_z[0:2, 0:2], ca[np.ix_([0, 2], [0, 2])])
    assert np.array_equal(c_z_dot[0:2, 0:2], ca[np.ix_([1, 3], [1, 3])])

    # Tracklets are independent, so the cross blocks stay zero.
    assert np.array_equal(c_z[0:2, 2:4], np.zeros((2, 2)))
    assert np.array_equal(c_z_dot[0:2, 2:4], np.zeros((2, 2)))


# --------------------------------------------------------------------------
# algebraic identities the weighted loss must satisfy
# --------------------------------------------------------------------------

def test_weighted_loss_matches_an_independent_computation():
    """Rebuild Eq. (18) with solve() instead of inv() and compare.

    Recomputes C_z_hat_dot = J C_z J^T here too, so a transpose error in
    the module would show up as a disagreement.
    """
    geometry = build_geometry()
    ca = realistic_covariance()

    from_module = weighted(geometry, ca, ca)[0]

    z, z_dot, residuals = rate_residuals(*loss_arguments(geometry))
    jacobian = z_hat_dot_jacobian(
        z, geometry["rho_1"], geometry["rho_2"], geometry["k"],
        geometry["r_s1"], geometry["v_s1"], geometry["t1"],
        geometry["r_s2"], geometry["v_s2"], geometry["t2"],
    )[0]

    combined = (
        measured_z_dot_covariance(ca, ca)
        + jacobian @ measured_z_covariance(ca, ca) @ jacobian.T
    )
    by_hand = float(residuals[0] @ np.linalg.solve(combined, residuals[0]))

    assert np.isclose(from_module, by_hand, rtol=1e-9, atol=0.0), (
        from_module, by_hand
    )


def test_loss_scales_inversely_with_covariance():
    """L is a quadratic form in C^-1, so scaling Ca by s must scale L by 1/s."""
    geometry = build_geometry()
    ca = realistic_covariance()

    reference = weighted(geometry, ca, ca)[0]

    for scale in [2.0, 10.0, 100.0]:
        scaled = weighted(geometry, scale * ca, scale * ca)[0]

        assert np.isclose(scaled * scale, reference, rtol=1e-9, atol=0.0), (
            scale, scaled * scale, reference
        )


def test_identity_covariance_reduces_to_squared_residual():
    """With C_z = 0 and C_z_dot = c*I, Eq. (18) collapses to |residual|^2 / c.

    Zeroing the angle block kills the Eq. (19) term, leaving the weighting
    entirely to C_z_dot -- so this checks the quadratic form itself,
    independent of the Jacobian.
    """
    geometry = build_geometry()
    _, _, residuals = rate_residuals(*loss_arguments(geometry))
    squared = float(residuals[0] @ residuals[0])

    for c in [1.0, 0.5, 4.0]:
        ca = np.zeros((4, 4))
        ca[1, 1] = ca[3, 3] = c  # rate block c*I, angle block zero

        assert np.isclose(weighted(geometry, ca, ca)[0], squared / c,
                          rtol=1e-9, atol=0.0), c


def test_zero_covariance_makes_the_weighted_loss_singular():
    """Why discriminator_loss_unweighted exists.

    The clean chain's Ca is the zero matrix, so Eq. (18) would invert a
    singular matrix. This pins that down rather than leaving it as a
    remark in a docstring.
    """
    geometry = build_geometry()
    zero = np.zeros((4, 4))

    try:
        weighted(geometry, zero, zero)
    except np.linalg.LinAlgError:
        return
    raise AssertionError("zero covariance should have produced a singular matrix")


# --------------------------------------------------------------------------
# the unweighted loss
# --------------------------------------------------------------------------

def test_unweighted_is_the_squared_residual_in_arcsec():
    """L = |residual|^2, converted rad/s -> arcsec/s. Units only, no weighting."""
    geometry = build_geometry()

    from_module = discriminator_loss_unweighted(*loss_arguments(geometry))[0]

    _, _, residuals = rate_residuals(*loss_arguments(geometry))
    in_radians = float(residuals[0] @ residuals[0])

    assert np.isclose(from_module, in_radians * _RAD_TO_ARCSEC**2,
                      rtol=1e-12, atol=0.0)


def test_both_losses_score_the_same_residual():
    """The two variants must differ only in weighting, never in the residual."""
    geometry = build_geometry()
    ca = np.zeros((4, 4))
    ca[1, 1] = ca[3, 3] = 1.0  # C_z_dot = I, C_z = 0

    unweighted = discriminator_loss_unweighted(*loss_arguments(geometry))
    weighted_with_identity = weighted(geometry, ca, ca)

    assert len(unweighted) == len(weighted_with_identity)

    for u, w in zip(unweighted, weighted_with_identity):
        assert np.isclose(u, w * _RAD_TO_ARCSEC**2, rtol=1e-9, atol=0.0), (u, w)


def test_evaluate_loss_dispatches_by_variant():
    """The clean chain must get the unweighted loss, the noisy chain Eq. (18)."""
    geometry = build_geometry()
    ca = realistic_covariance()
    g = geometry

    assert LOSS_FUNCTIONS["clean"] is discriminator_loss_unweighted
    assert LOSS_FUNCTIONS["noisy"] is discriminator_loss

    clean = evaluate_loss(
        "clean", g["rho_1"], g["rho_2"], g["k"],
        g["r_s1"], g["v_s1"], g["t1"], g["attributable_1"], ca,
        g["r_s2"], g["v_s2"], g["t2"], g["attributable_2"], ca,
    )
    noisy = evaluate_loss(
        "noisy", g["rho_1"], g["rho_2"], g["k"],
        g["r_s1"], g["v_s1"], g["t1"], g["attributable_1"], ca,
        g["r_s2"], g["v_s2"], g["t2"], g["attributable_2"], ca,
    )

    assert np.allclose(clean, discriminator_loss_unweighted(*loss_arguments(geometry)))
    assert np.allclose(noisy, weighted(geometry, ca, ca))

    # The clean call must ignore the covariance it was handed.
    other = evaluate_loss(
        "clean", g["rho_1"], g["rho_2"], g["k"],
        g["r_s1"], g["v_s1"], g["t1"], g["attributable_1"], 99.0 * ca,
        g["r_s2"], g["v_s2"], g["t2"], g["attributable_2"], 99.0 * ca,
    )
    assert np.allclose(clean, other)


# --------------------------------------------------------------------------
# the Eq. (19) Jacobian
# --------------------------------------------------------------------------

def test_jacobian_is_stable_under_step_halving():
    """Halving eps should barely move a converged finite-difference Jacobian.

    A consistency check only -- unlike model_rates.py there is no
    independent closed form to compare against.
    """
    geometry = build_geometry()
    z, _, _ = rate_residuals(*loss_arguments(geometry))

    def jacobian_at(eps):
        return z_hat_dot_jacobian(
            z, geometry["rho_1"], geometry["rho_2"], geometry["k"],
            geometry["r_s1"], geometry["v_s1"], geometry["t1"],
            geometry["r_s2"], geometry["v_s2"], geometry["t2"],
            eps=eps,
        )[0]

    coarse = jacobian_at(1e-6)
    fine = jacobian_at(5e-7)

    relative_change = np.max(np.abs(coarse - fine)) / np.max(np.abs(coarse))

    assert relative_change < 1e-3, relative_change


def test_jacobian_has_one_entry_per_candidate():
    """Jacobians must line up with candidates by position, or losses mismatch."""
    geometry = build_geometry()
    z, _, residuals = rate_residuals(*loss_arguments(geometry))

    jacobians = z_hat_dot_jacobian(
        z, geometry["rho_1"], geometry["rho_2"], geometry["k"],
        geometry["r_s1"], geometry["v_s1"], geometry["t1"],
        geometry["r_s2"], geometry["v_s2"], geometry["t2"],
    )

    assert len(jacobians) == len(residuals)
    assert all(j.shape == (4, 4) for j in jacobians)


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

TESTS = [
    test_z_and_z_dot_ordering,
    test_covariance_sub_blocks,
    test_weighted_loss_matches_an_independent_computation,
    test_loss_scales_inversely_with_covariance,
    test_identity_covariance_reduces_to_squared_residual,
    test_zero_covariance_makes_the_weighted_loss_singular,
    test_unweighted_is_the_squared_residual_in_arcsec,
    test_both_losses_score_the_same_residual,
    test_evaluate_loss_dispatches_by_variant,
    test_jacobian_is_stable_under_step_halving,
    test_jacobian_has_one_entry_per_candidate,
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
