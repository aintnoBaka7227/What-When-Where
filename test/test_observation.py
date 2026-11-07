"""Tests for stage 2, fit_attributes.py and create_attributes_vector.py.

    uv run python test/test_observation.py

A linear least-squares fit, so: hand-worked cases, edge cases, then generated
data. See TESTING.md.

For 5 points 20 s apart, centred times are -40..40 s, X^T X = [[5, 0],
[0, 4000]], so var(angle) = s^2/5 + s_sys^2, var(rate) = s^2/4000, and the
angle-rate covariance is 0.
"""

import numpy as np

from lab.data.dataset import (
    NOISY_VARIANTS,
    RA_COLUMN,
    VARIANTS,
    data_file,
    load_data,
)
from lab.data.gcrs_data_generation import gcrs_to_radec
from lab.model.model_rates import angle_rate_from_state
from lab.observation.create_attributes_vector import (
    ATTRIBUTABLES_NPZ_FILES,
    ATTRIBUTABLES_TXT_FILES,
    build_attributables,
)
from lab.observation.fit_attributes import fit_attributes
from pipeline_fixture import pipeline, run_tests

ARCSEC = np.deg2rad(1.0 / 3600.0)

# The generator's tracklet sampling.
TIMES = np.array([0.0, 20.0, 40.0, 60.0, 80.0]) + 50_000.0


# --------------------------------------------------------------------------
# 1. simple cases
# --------------------------------------------------------------------------

def test_exact_line_is_recovered():
    alpha_0, alpha_rate = 1.2, 7.3e-5
    delta_0, delta_rate = -0.1, -2.1e-6
    t0 = TIMES.mean()
    ra = alpha_0 + alpha_rate * (TIMES - t0)
    dec = delta_0 + delta_rate * (TIMES - t0)

    a, _, fitted_t0 = fit_attributes(TIMES, ra, dec, 0.7, 5.0)

    assert fitted_t0 == t0
    assert np.allclose(a, [alpha_0, alpha_rate, delta_0, delta_rate], rtol=0.0, atol=1e-15), a


def test_constant_angle_gives_zero_rate():
    a, _, _ = fit_attributes(TIMES, np.full(5, 3.0), np.full(5, 0.2), 0.7, 5.0)
    assert np.allclose(a, [3.0, 0.0, 0.2, 0.0], rtol=0.0, atol=1e-15), a


def test_covariance_matches_the_hand_formula():
    sigma, sigma_sys = 1.2, 5.0
    _, ca, _ = fit_attributes(TIMES, np.zeros(5), np.zeros(5), sigma, sigma_sys)

    s, s_sys = sigma * ARCSEC, sigma_sys * ARCSEC
    block = np.array([[s**2 / 5 + s_sys**2, 0.0], [0.0, s**2 / 4000.0]])
    expected = np.zeros((4, 4))
    expected[0:2, 0:2] = block
    expected[2:4, 2:4] = block

    assert np.allclose(ca, expected, rtol=1e-12, atol=0.0), ca


def test_systematic_error_does_not_reach_the_rates():
    """A constant offset moves the fitted angle, never the fitted rate."""
    _, without, _ = fit_attributes(TIMES, np.zeros(5), np.zeros(5), 0.7, 0.0)
    _, with_sys, _ = fit_attributes(TIMES, np.zeros(5), np.zeros(5), 0.7, 5.0)

    rates = [1, 3]
    assert np.array_equal(with_sys[np.ix_(rates, rates)], without[np.ix_(rates, rates)])
    assert with_sys[0, 0] > without[0, 0] and with_sys[2, 2] > without[2, 2]


def test_a_constant_offset_shifts_only_the_angle():
    """What the systematic term models: a bias moves alpha, not alpha_dot."""
    ra = 1.0 + 3e-5 * (TIMES - TIMES.mean())
    a, _, _ = fit_attributes(TIMES, ra, np.zeros(5), 0.7, 5.0)
    shifted, _, _ = fit_attributes(TIMES, ra + 5 * ARCSEC, np.zeros(5), 0.7, 5.0)

    assert np.isclose(shifted[0] - a[0], 5 * ARCSEC, rtol=1e-9, atol=0.0)
    assert np.isclose(shifted[1], a[1], rtol=1e-9, atol=1e-20)


# --------------------------------------------------------------------------
# 2. edge cases
# --------------------------------------------------------------------------

def test_ra_crossing_two_pi_keeps_a_continuous_rate():
    """The noisy generator writes RA % 2*pi, so a tracklet can wrap to 0."""
    rate = 7.3e-5
    true_ra = 2 * np.pi - 2e-3 + rate * (TIMES - TIMES[0])
    wrapped = true_ra % (2 * np.pi)
    assert wrapped[0] > 6.0 and wrapped[-1] < 0.1, "test does not actually wrap"

    a, _, _ = fit_attributes(TIMES, wrapped, np.zeros(5), 0.7, 5.0)

    assert np.isclose(a[1], rate, rtol=1e-9), a[1]
    # alpha may land outside [0, 2pi) after unwrapping; it must be the same angle.
    expected = true_ra.mean() % (2 * np.pi)
    assert np.isclose(a[0] % (2 * np.pi), expected, rtol=0.0, atol=1e-12)


def test_shuffled_and_wrapping_rows_give_the_same_attributable():
    """Row order must not matter, even across RA = 2*pi.

    All RA is rotated so the first tracklet straddles 2*pi (rates unchanged),
    then the rows are shuffled.
    """
    with pipeline(num_objects=1, stages=1):
        build_attributables("clean", ATTRIBUTABLES_NPZ_FILES["clean"], ATTRIBUTABLES_TXT_FILES["clean"])
        before = dict(np.load(ATTRIBUTABLES_NPZ_FILES["clean"]))

        data = load_data("clean")
        shift = 2 * np.pi - data[2, RA_COLUMN]
        data[:, RA_COLUMN] = (data[:, RA_COLUMN] + shift) % (2 * np.pi)
        first = data[:5, RA_COLUMN]
        assert first.max() > 6.0 and first.min() < 0.3, "first tracklet does not wrap"

        rng = np.random.default_rng(0)
        np.savetxt(data_file("clean"), data[rng.permutation(len(data))],
                   delimiter=",", fmt="%.17g")
        build_attributables("clean", ATTRIBUTABLES_NPZ_FILES["clean"], ATTRIBUTABLES_TXT_FILES["clean"])
        after = dict(np.load(ATTRIBUTABLES_NPZ_FILES["clean"]))

    for key in ("object_id", "tracklet_id", "t0", "Ca"):
        assert np.array_equal(before[key], after[key]), key
    rates = [1, 3]
    assert np.allclose(before["attributable"][:, rates], after["attributable"][:, rates],
                       rtol=1e-9, atol=0.0)
    d_alpha = (after["attributable"][:, 0] - before["attributable"][:, 0] - shift) % (2 * np.pi)
    assert np.allclose(np.minimum(d_alpha, 2 * np.pi - d_alpha), 0.0, atol=1e-9)


def test_two_observations_are_the_minimum():
    times = np.array([0.0, 20.0])
    a, ca, t0 = fit_attributes(times, [1.0, 1.001], [0.0, -0.002], 0.7, 0.0)

    assert t0 == 10.0
    assert np.allclose(a, [1.0005, 0.001 / 20, -0.001, -0.002 / 20], rtol=1e-12)
    # Two points fit exactly, but the covariance is still defined: var(rate) = 2 s^2 / 400.
    assert np.isclose(ca[1, 1], 2 * (0.7 * ARCSEC) ** 2 / 400.0, rtol=1e-12)


def test_a_single_observation_cannot_give_a_rate():
    try:
        fit_attributes([0.0], [1.0], [0.0], 0.7, 5.0)
    except np.linalg.LinAlgError:
        return
    raise AssertionError("one observation produced an attributable")


def test_uneven_spacing_uses_the_mean_time():
    times = np.array([0.0, 10.0, 70.0])
    _, ca, t0 = fit_attributes(times, np.zeros(3), np.zeros(3), 0.7, 0.0)
    assert np.isclose(t0, 80.0 / 3.0)
    # Centring on the mean still decouples angle and rate.
    assert abs(ca[0, 1]) < 1e-12 * ca[0, 0]


def test_zero_uncertainty_gives_the_zero_matrix():
    """The clean chain: exact attributable, zero covariance. Stage 7 relies on it."""
    _, ca, _ = fit_attributes(TIMES, np.zeros(5), np.zeros(5), 0.0, 0.0)
    assert np.array_equal(ca, np.zeros((4, 4)))


# --------------------------------------------------------------------------
# 3. on the generator's data
# --------------------------------------------------------------------------

def test_one_attributable_per_tracklet_in_every_chain():
    with pipeline(num_objects=3, stages=2) as run:
        clean = run.attributables("clean")
        for variant in VARIANTS:
            other = run.attributables(variant)
            assert len(other["t0"]) == 15, variant
            # Same tracklets, same epochs, row for row.
            assert np.array_equal(other["object_id"], clean["object_id"])
            assert np.array_equal(other["tracklet_id"], clean["tracklet_id"])
            assert np.array_equal(other["t0"], clean["t0"])


def test_clean_rates_equal_the_true_rates():
    """Without noise, fitted rates equal the true rates at t0.

    Symmetric sampling cancels the quadratic term, so the rate is near exact;
    the angle keeps a ~0.2" curvature bias.
    """
    with pipeline(num_objects=4, stages=2) as run:
        clean = run.attributables("clean")
        worst_rate = worst_angle = 0.0
        for n, (obj, trk) in enumerate(zip(clean["object_id"], clean["tracklet_id"])):
            _, r, v, r_site, v_site = run.truth_at_epoch(obj, trk)
            alpha_dot, delta_dot = angle_rate_from_state(r - r_site, v - v_site)
            alpha, delta = gcrs_to_radec(r - r_site)

            fitted = clean["attributable"][n]
            worst_rate = max(worst_rate, abs(fitted[1] - alpha_dot), abs(fitted[3] - delta_dot))
            d_alpha = (fitted[0] - alpha + np.pi) % (2 * np.pi) - np.pi
            worst_angle = max(worst_angle, abs(d_alpha), abs(fitted[2] - delta))

    assert worst_rate < 1e-9, f"rate error {worst_rate:.2e} rad/s"
    assert worst_angle < 0.5 * ARCSEC, f"angle error {worst_angle / ARCSEC:.3f} arcsec"


def test_noisy_minus_clean_follows_the_covariance():
    """Noisy - clean is the fit of the noise, so d^T Ca^-1 d ~ chi-squared(4).

    A mean far from 4 means Ca is misscaled or missing a term.
    """
    with pipeline(num_objects=10, stages=2) as run:
        clean = run.attributables("clean")["attributable"]
        for variant in NOISY_VARIANTS:
            noisy = run.attributables(variant)
            d = noisy["attributable"] - clean
            d[:, 0] = (d[:, 0] + np.pi) % (2 * np.pi) - np.pi
            distances = np.array([x @ np.linalg.solve(c, x) for x, c in zip(d, noisy["Ca"])])

            # 50 tracklets: the mean of chi-squared(4) has sd sqrt(8/50) = 0.4.
            assert 2.4 < distances.mean() < 5.6, (variant, distances.mean())


def test_rate_noise_scales_exactly_with_sigma():
    """The generator shares one unit draw across noise levels and adds the
    same bias to all, so rate errors divided by sigma must agree exactly.
    (The bias is constant within a tracklet and cancels out of the rate.)
    """
    with pipeline(num_objects=3, stages=2) as run:
        clean = run.attributables("clean")["attributable"][:, [1, 3]]
        scaled = [
            (run.attributables(variant)["attributable"][:, [1, 3]] - clean) / sigma
            for variant, sigma in NOISY_VARIANTS.items()
        ]

    for other in scaled[1:]:
        assert np.allclose(other, scaled[0], rtol=1e-6, atol=0.0)


def test_noisy_ra_is_wrapped_but_fits_unwrapped():
    """Adherence check: the generator wraps noisy RA into [0, 2pi)."""
    with pipeline(num_objects=3, stages=1):
        for variant in NOISY_VARIANTS:
            ra = load_data(variant)[:, RA_COLUMN]
            assert np.all((ra >= 0.0) & (ra < 2 * np.pi)), variant


TESTS = [
    test_exact_line_is_recovered,
    test_constant_angle_gives_zero_rate,
    test_covariance_matches_the_hand_formula,
    test_systematic_error_does_not_reach_the_rates,
    test_a_constant_offset_shifts_only_the_angle,
    test_ra_crossing_two_pi_keeps_a_continuous_rate,
    test_shuffled_and_wrapping_rows_give_the_same_attributable,
    test_two_observations_are_the_minimum,
    test_a_single_observation_cannot_give_a_rate,
    test_uneven_spacing_uses_the_mean_time,
    test_zero_uncertainty_gives_the_zero_matrix,
    test_one_attributable_per_tracklet_in_every_chain,
    test_clean_rates_equal_the_true_rates,
    test_noisy_minus_clean_follows_the_covariance,
    test_rate_noise_scales_exactly_with_sigma,
    test_noisy_ra_is_wrapped_but_fits_unwrapped,
]


if __name__ == "__main__":
    raise SystemExit(run_tests(TESTS))
