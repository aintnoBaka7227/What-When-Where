"""Tests for src/lab/scoring/discriminator_loss.py.

    uv run python test/test_discriminator_loss.py

A wrong loss still returns plausible numbers, so most tests are identities
that hold at any feasible point:

    scaling      Ca -> s Ca gives L -> L / s
    reduction    C = c I gives L = |residual|^2 / c
    agreement    an independent solve() recomputation matches
    units        the unweighted loss is |residual|^2 * 206265^2

Then the two departures from the printed paper and the generated data:

    Eq. (19)     J C_z J^T matches a Monte Carlo covariance; J^T C_z J does not
    calibration  at the truth, clean L ~ 0 and noisy L ~ chi-squared(4)
    contrast     moving off the truth raises L
"""


import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_fixture import pipeline  # noqa: E402
from lab.data.dataset import NOISY_VARIANTS, VARIANTS  # noqa: E402
from lab.evaluation.metrics import true_k  # noqa: E402
from lab.model.lambert import LambertNoSolution  # noqa: E402
from lab.scoring.discriminator_loss import z_hat_dot_candidates_from_z  # noqa: E402

from lab.scoring.discriminator_loss import (
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
from lab.model.model_rates import angle_rate_from_state
from lab.data.gcrs_data_generation import two_body

EARTH_ROTATION_RATE = 7.292115e-5  # rad/s
EARTH_RADIUS_KM = 6378.137


def build_geometry():
    """A feasible two-tracklet hypothesis from a known near-GEO orbit.

    The identities hold at any feasible point, so it need not be a good one.
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
    """A Ca shaped like fit_attributes(): systematic term on the angles only.

    Non-zero angle-rate terms, since a diagonal Ca would hide a transpose error.
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
    """Eq. (18) rebuilt by hand, including J C_z J^T, must match."""
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
    """C_z = 0 and C_z_dot = c I give |residual|^2 / c.

    Zeroing C_z removes the Jacobian term, isolating the quadratic form.
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
    """Why the unweighted loss exists: clean's zero Ca makes Eq. (18) singular."""
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
    assert LOSS_FUNCTIONS["noisy_0.7as"] is discriminator_loss

    clean = evaluate_loss(
        "clean", g["rho_1"], g["rho_2"], g["k"],
        g["r_s1"], g["v_s1"], g["t1"], g["attributable_1"], ca,
        g["r_s2"], g["v_s2"], g["t2"], g["attributable_2"], ca,
    )
    noisy = evaluate_loss(
        "noisy_0.7as", g["rho_1"], g["rho_2"], g["k"],
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
    """Halving eps barely moves the Jacobian. There is no closed form to compare with."""
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
# Eq. (19): the covariance propagation, checked by Monte Carlo
# --------------------------------------------------------------------------

def test_jacobian_propagation_matches_monte_carlo():
    """Sample z ~ N(z, C_z); the covariance of z_hat_dot must match J C_z J^T.

    Uses only the forward model. The printed J^T C_z J must disagree by more
    than sampling error.
    """
    g = build_geometry()
    ca = realistic_covariance()
    z, _, _ = rate_residuals(*loss_arguments(g))
    c_z = measured_z_covariance(ca, ca)
    args = (g["rho_1"], g["rho_2"], g["k"], g["r_s1"], g["v_s1"], g["t1"],
            g["r_s2"], g["v_s2"], g["t2"])

    jacobian = z_hat_dot_jacobian(z, *args)[0]
    coded = jacobian @ c_z @ jacobian.T
    printed = jacobian.T @ c_z @ jacobian

    rng = np.random.default_rng(11)
    samples = np.array([
        z_hat_dot_candidates_from_z(rng.multivariate_normal(z, c_z), *args)[0]
        for _ in range(3000)
    ])
    sampled = np.cov(samples, rowvar=False)

    scale = np.max(np.abs(sampled))
    coded_error = np.max(np.abs(coded - sampled)) / scale
    printed_error = np.max(np.abs(printed - sampled)) / scale

    # 3000 samples: covariance entries carry about sqrt(2/3000) = 3% noise.
    assert coded_error < 0.1, coded_error
    assert printed_error > 3 * coded_error, (coded_error, printed_error)


# --------------------------------------------------------------------------
# on the generator's data: the loss at the truth
# --------------------------------------------------------------------------

def _true_pair_losses(run, variant, rho_offset_km=0.0):
    """The loss of every same-object pair at its true ranges and true k."""
    attributables = run.attributables(variant)
    rows = list(zip(attributables["object_id"], attributables["tracklet_id"]))
    losses = []
    for i, j in combinations(range(len(rows)), 2):
        if rows[i][0] != rows[j][0]:
            continue
        t1, r1, v1, r_s1, v_s1 = run.truth_at_epoch(*rows[i])
        t2, r2, _, r_s2, v_s2 = run.truth_at_epoch(*rows[j])
        try:
            values = evaluate_loss(
                variant,
                np.linalg.norm(r1 - r_s1) + rho_offset_km, np.linalg.norm(r2 - r_s2),
                true_k(r1, v1, r2, t2 - t1),
                r_s1, v_s1, t1, attributables["attributable"][i], attributables["Ca"][i],
                r_s2, v_s2, t2, attributables["attributable"][j], attributables["Ca"][j],
            )
        except LambertNoSolution:
            # Off the truth a multi-revolution k can become infeasible; Search
            # scores that as its penalty, so it counts as a high loss here.
            values = [np.inf]
        losses.append(min(values))
    return np.array(losses)


def test_clean_loss_is_zero_at_the_truth():
    """No noise, the true orbit: any residual is a code bug."""
    with pipeline(num_objects=5, stages=2) as run:
        losses = _true_pair_losses(run, "clean")
    assert len(losses) == 50
    # arcsec^2/s^2. A 1e-3 "/s rate error alone would score 1e-6.
    assert losses.max() < 1e-6, losses.max()


def test_noisy_loss_at_the_truth_is_chi_squared_4():
    """Section 5.2's claim: L at the truth is chi-squared(4), mean 4.

    Pairs share tracklets, so the band is wide. A missing or misscaled
    covariance term moves the mean.
    """
    with pipeline(num_objects=10, stages=2) as run:
        for variant in NOISY_VARIANTS:
            losses = _true_pair_losses(run, variant)
            assert 2.5 < losses.mean() < 6.0, (variant, losses.mean())
            # chi-squared(4) exceeds 18.5 one time in 1000.
            assert np.mean(losses > 18.5) < 0.03, (variant, np.sort(losses)[-5:])


def test_loss_rises_away_from_the_truth():
    with pipeline(num_objects=3, stages=2) as run:
        for variant in VARIANTS:
            at_truth = _true_pair_losses(run, variant)
            offset = _true_pair_losses(run, variant, rho_offset_km=300.0)
            assert np.median(offset) > 10 * np.median(at_truth), variant


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
    test_jacobian_propagation_matches_monte_carlo,
    test_clean_loss_is_zero_at_the_truth,
    test_noisy_loss_at_the_truth_is_chi_squared_4,
    test_loss_rises_away_from_the_truth,
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
        except Exception as exc:
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
