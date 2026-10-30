"""Validation suite for the station velocity in src/lab/data/gcrs_data_generation.py.

Run directly:

    uv run python test/test_station_velocity.py

The generator places each station by attaching a GCRS point to the rotating
Earth (`fix_station_to_earth`, via ITRS) and then reads the station's GCRS
position and velocity at every observation (`station_gcrs_state`). That
velocity is the r_s-dot term of Siminski et al. (2014) Eq. (6), so an error in
it goes straight into the modelled angular rates.

Nothing here computes an expected value by calling the code under test. Each
check compares astropy's velocity against something derived independently:

  1. Rigid rotation      |v| = omega_E * (distance from Earth's spin axis),
                         using the station's fixed ITRS coordinates and the
                         IERS nominal rotation rate. Catches unit errors
                         (m vs km), a missing velocity, or a wrong magnitude.
  2. Geometry            v is horizontal (v . r = 0), eastward, and lies along
                         omega x r. Catches sign and orientation errors.
  3. Position closure    v equals the finite difference of the station's own
                         positions. Ties the velocity to the positions the
                         generator writes, which is what Eq. (6) assumes.
  4. Use in Eq. (6)      Topocentric RA/Dec rates built from v_site match the
                         finite-difference rates of the observed angles, and
                         would not match without it.
  5. Generated data      The same physical checks on rows from the generator,
                         written to a temporary folder.

Edge cases: stations at the poles (speed ~0), the equator (maximum speed),
the antimeridian, an elevated station, the geocentre, a time step across the
2012-06-30 leap second, and a full sidereal day of rotation.
"""

import tempfile
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

import lab.paths
from lab.data.dataset import load_data
from lab.data.gcrs_data_generation import (
    fix_station_to_earth,
    generate_object,
    station_gcrs_state,
    two_body,
    mu as MU,
    NUM_OF_OUTPUT_COLUMNS,
)


# IERS nominal mean angular velocity of the Earth, rad/s. Independent of
# astropy's own Earth-orientation model; length-of-day variations change the
# true rate by ~1e-8 relative, far inside the tolerances below.
OMEGA_EARTH = 7.292115e-5

# Epoch the generator uses for measurement_time = 0.
EPOCH = Time("2012-08-18T22:59:08", scale="utc")

# Zimmerwald (ZimSMART), the station behind the paper's data set.
ZIMMERWALD = EarthLocation.from_geodetic(
    lon=7.465 * u.deg, lat=46.877 * u.deg, height=951.0 * u.m
)

# Relative agreement required between astropy's speed and omega_E * rho.
# The reference measures rho from the ITRS z axis, but Earth actually spins
# about the CIP, which polar motion tilts ~0.3" (1.5e-6 rad) from ITRS z. That
# shifts rho by ~R * 1.5e-6 * sin(lat), a relative error ~1.5e-6 * tan(lat):
# measured 2e-8 at the equator, 2e-6 at 45 deg, 1.2e-5 at 80 deg. The
# tolerance follows that, with headroom; a unit or sign error is off by
# orders of magnitude.
SPEED_REL_TOL = 1.0e-6
POLAR_MOTION_RAD = 3.0e-6

# Finite-difference half step, s. The truncation error (omega*dt)^2 / 6 is
# ~1e-9 relative at 1 s, below float64 noise on positions differenced at
# Earth-radius scale.
FD_HALF_STEP_S = 1.0
FD_REL_TOL = 1.0e-6

# The GCRS z axis and Earth's spin axis differ by precession-nutation, about
# 0.17 deg in 2012, so the direction check against omega_E z-hat x r allows
# 0.5 deg. A reversed or rotated velocity is off by tens of degrees or 180.
DIRECTION_TOL_DEG = 0.5


# ----------------------------------------------------------------------------
# Independent references
# ----------------------------------------------------------------------------

def axis_distance_km(station):
    """Distance from Earth's spin axis, from the station's fixed ITRS coordinates."""
    x = station.x.to_value(u.km)
    y = station.y.to_value(u.km)
    return np.hypot(x, y)


def expected_speed_km_s(station):
    """Speed of a point rigidly rotating with the Earth."""
    return OMEGA_EARTH * axis_distance_km(station)


def speed_tolerance(station):
    """Relative speed tolerance, widened with latitude for polar motion."""
    return SPEED_REL_TOL + POLAR_MOTION_RAD * abs(np.tan(station.lat.to_value(u.rad)))


def finite_difference_velocity(station, time, half_step_s=FD_HALF_STEP_S):
    """Central difference of the station's GCRS positions around time."""
    r_plus, _ = station_gcrs_state(station, time + half_step_s * u.s)
    r_minus, _ = station_gcrs_state(station, time - half_step_s * u.s)
    return (r_plus - r_minus) / (2.0 * half_step_s)


def relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def angle_between_deg(a, b):
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def radec(line_of_sight):
    x, y, z = line_of_sight
    return np.arctan2(y, x), np.arcsin(z / np.linalg.norm(line_of_sight))


def radec_rates(line_of_sight, line_of_sight_rate):
    """Analytic RA/Dec rates from a line of sight and its time derivative."""
    x, y, z = line_of_sight
    xd, yd, zd = line_of_sight_rate
    rho_xy_sq = x * x + y * y
    rho_sq = rho_xy_sq + z * z
    ra_rate = (x * yd - y * xd) / rho_xy_sq
    dec_rate = (zd - z * np.dot(line_of_sight, line_of_sight_rate) / rho_sq) / np.sqrt(rho_xy_sq)
    return np.array([ra_rate, dec_rate])


# ----------------------------------------------------------------------------
# 1-3. Normal cases: a mid-latitude station
# ----------------------------------------------------------------------------

def test_units_are_km_and_km_per_s():
    r_site, v_site = station_gcrs_state(ZIMMERWALD, EPOCH)
    radius = np.linalg.norm(r_site)
    speed = np.linalg.norm(v_site)
    assert 6350.0 < radius < 6400.0, f"|r_site| = {radius} -- expected km near Earth's radius"
    assert 0.0 < speed < 0.47, f"|v_site| = {speed} -- expected km/s below omega_E * R_eq"


def test_speed_matches_earth_rotation():
    _, v_site = station_gcrs_state(ZIMMERWALD, EPOCH)
    speed = np.linalg.norm(v_site)
    expected = expected_speed_km_s(ZIMMERWALD)
    error = abs(speed - expected) / expected
    assert error < speed_tolerance(ZIMMERWALD), (
        f"|v| = {speed:.9f} km/s, omega_E * rho = {expected:.9f} km/s, rel err {error:.2e}"
    )


def test_velocity_is_horizontal():
    r_site, v_site = station_gcrs_state(ZIMMERWALD, EPOCH)
    radial = np.dot(v_site, r_site) / np.linalg.norm(r_site)
    assert abs(radial) < 1.0e-9, f"radial velocity {radial:.3e} km/s -- should be zero"


def test_velocity_is_eastward_along_omega_cross_r():
    r_site, v_site = station_gcrs_state(ZIMMERWALD, EPOCH)
    expected_direction = np.cross([0.0, 0.0, OMEGA_EARTH], r_site)
    angle = angle_between_deg(v_site, expected_direction)
    assert angle < DIRECTION_TOL_DEG, (
        f"velocity is {angle:.3f} deg from omega x r -- expected eastward rotation"
    )


def test_velocity_matches_finite_difference_of_position():
    _, v_site = station_gcrs_state(ZIMMERWALD, EPOCH)
    v_fd = finite_difference_velocity(ZIMMERWALD, EPOCH)
    error = relative_error(v_site, v_fd)
    assert error < FD_REL_TOL, f"v_site vs finite difference: rel err {error:.2e}"


def test_velocity_is_consistent_across_the_generator_span():
    # The generator observes over 0-52 h. Speed must stay at omega_E * rho and
    # the velocity must keep matching its positions throughout.
    expected = expected_speed_km_s(ZIMMERWALD)
    for hours in (0.0, 13.0, 26.0, 39.0, 52.0):
        time = EPOCH + hours * u.hour
        _, v_site = station_gcrs_state(ZIMMERWALD, time)
        speed_error = abs(np.linalg.norm(v_site) - expected) / expected
        fd_error = relative_error(v_site, finite_difference_velocity(ZIMMERWALD, time))
        assert speed_error < speed_tolerance(ZIMMERWALD), f"t = {hours} h: speed rel err {speed_error:.2e}"
        assert fd_error < FD_REL_TOL, f"t = {hours} h: finite-difference rel err {fd_error:.2e}"


def test_fix_station_to_earth_reproduces_the_chosen_point():
    # The generator picks a GCRS point (in meters) and attaches it to the
    # ground. At the attachment time the station must be at that point.
    chosen_m = np.array([4.1e6, -3.2e6, 3.7e6])
    chosen_m *= 6378.1e3 / np.linalg.norm(chosen_m)
    station = fix_station_to_earth(chosen_m, EPOCH)
    r_site, _ = station_gcrs_state(station, EPOCH)
    miss_m = np.linalg.norm(r_site * 1000.0 - chosen_m)
    assert miss_m < 1.0e-3, f"station is {miss_m:.3e} m from the chosen point"


def test_fix_station_to_earth_gives_a_rotating_station():
    # The ITRS round trip is what makes the station move; a station left in
    # GCRS would have zero velocity. Check the attached station's velocity
    # against rigid rotation and against its own positions.
    chosen_m = np.array([-2.0e6, 5.5e6, 2.5e6])
    chosen_m *= 6378.1e3 / np.linalg.norm(chosen_m)
    station = fix_station_to_earth(chosen_m, EPOCH)
    time = EPOCH + 40.0 * u.s
    _, v_site = station_gcrs_state(station, time)
    expected = expected_speed_km_s(station)
    speed_error = abs(np.linalg.norm(v_site) - expected) / expected
    fd_error = relative_error(v_site, finite_difference_velocity(station, time))
    assert speed_error < speed_tolerance(station), f"speed rel err {speed_error:.2e}"
    assert fd_error < FD_REL_TOL, f"finite-difference rel err {fd_error:.2e}"


# ----------------------------------------------------------------------------
# 4. Use in Eq. (6): the velocity produces the right topocentric rates
# ----------------------------------------------------------------------------

def _geo_satellite_above(station, time):
    """A circular, inclined GEO-radius orbit passing over the station at time."""
    r_site, _ = station_gcrs_state(station, time)
    r_hat = r_site / np.linalg.norm(r_site)
    r_sat = 42164.0 * r_hat
    along = np.cross([0.0, 0.0, 1.0], r_hat)
    along /= np.linalg.norm(along)
    v_sat = np.sqrt(MU / 42164.0) * along
    return r_sat, v_sat


def _observed_rates(station, time, r_sat0, v_sat0):
    """Finite-difference RA/Dec rates from positions only."""
    angles = []
    for step in (-FD_HALF_STEP_S, FD_HALF_STEP_S):
        r_sat, _ = two_body(MU, step, r_sat0, v_sat0)
        r_site, _ = station_gcrs_state(station, time + step * u.s)
        angles.append(radec(r_sat - r_site))
    (ra_m, dec_m), (ra_p, dec_p) = angles
    d_ra = (ra_p - ra_m + np.pi) % (2.0 * np.pi) - np.pi
    return np.array([d_ra, dec_p - dec_m]) / (2.0 * FD_HALF_STEP_S)


def test_topocentric_rates_use_station_velocity_correctly():
    r_sat, v_sat = _geo_satellite_above(ZIMMERWALD, EPOCH)
    r_site, v_site = station_gcrs_state(ZIMMERWALD, EPOCH)
    line_of_sight = r_sat - r_site

    observed = _observed_rates(ZIMMERWALD, EPOCH, r_sat, v_sat)
    modelled = radec_rates(line_of_sight, v_sat - v_site)
    without_station = radec_rates(line_of_sight, v_sat)

    error = relative_error(modelled, observed)
    error_without = relative_error(without_station, observed)

    assert error < 1.0e-5, f"Eq. (6) rates with v_site: rel err {error:.2e}"
    # The check has power only if leaving v_site out would have failed it.
    assert error_without > 1.0e-2, (
        f"rates without v_site are within {error_without:.2e} -- test cannot "
        f"distinguish a missing station velocity"
    )


# ----------------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------------

def test_north_pole_station_is_almost_stationary():
    # On the spin axis rho ~ 0. Polar motion puts the ITRS pole a few meters
    # off the rotation axis, so the speed is ~1 mm/s, not exactly zero.
    pole = EarthLocation.from_geodetic(lon=0.0 * u.deg, lat=90.0 * u.deg, height=0.0 * u.m)
    _, v_site = station_gcrs_state(pole, EPOCH)
    speed = np.linalg.norm(v_site)
    assert speed < 1.0e-5, f"north-pole speed {speed:.3e} km/s -- should be ~0"
    assert np.all(np.isfinite(v_site)), "north-pole velocity is not finite"


def test_south_pole_station_is_almost_stationary():
    pole = EarthLocation.from_geodetic(lon=0.0 * u.deg, lat=-90.0 * u.deg, height=0.0 * u.m)
    _, v_site = station_gcrs_state(pole, EPOCH)
    speed = np.linalg.norm(v_site)
    assert speed < 1.0e-5, f"south-pole speed {speed:.3e} km/s -- should be ~0"
    assert np.all(np.isfinite(v_site)), "south-pole velocity is not finite"


def test_equator_station_has_maximum_speed():
    equator = EarthLocation.from_geodetic(lon=0.0 * u.deg, lat=0.0 * u.deg, height=0.0 * u.m)
    _, v_equator = station_gcrs_state(equator, EPOCH)
    speed = np.linalg.norm(v_equator)

    # omega_E * WGS84 equatorial radius.
    expected = OMEGA_EARTH * 6378.137
    error = abs(speed - expected) / expected
    assert error < SPEED_REL_TOL, f"equator speed {speed:.9f} vs {expected:.9f}, rel err {error:.2e}"

    _, v_mid = station_gcrs_state(ZIMMERWALD, EPOCH)
    assert speed > np.linalg.norm(v_mid), "equator should be faster than mid-latitude"


def test_speed_follows_cos_latitude():
    # Same longitude, latitudes 0..80 deg: speed must fall as the distance
    # from the spin axis does.
    for latitude in (0.0, 20.0, 45.0, 60.0, 80.0):
        station = EarthLocation.from_geodetic(
            lon=100.0 * u.deg, lat=latitude * u.deg, height=0.0 * u.m
        )
        _, v_site = station_gcrs_state(station, EPOCH)
        expected = expected_speed_km_s(station)
        error = abs(np.linalg.norm(v_site) - expected) / expected
        assert error < speed_tolerance(station), f"lat {latitude} deg: speed rel err {error:.2e}"


def test_southern_hemisphere_still_rotates_eastward():
    station = EarthLocation.from_geodetic(lon=-70.7 * u.deg, lat=-30.2 * u.deg, height=0.0 * u.m)
    r_site, v_site = station_gcrs_state(station, EPOCH)
    angle = angle_between_deg(v_site, np.cross([0.0, 0.0, OMEGA_EARTH], r_site))
    assert angle < DIRECTION_TOL_DEG, f"southern station velocity {angle:.3f} deg from omega x r"


def test_antimeridian_longitudes_agree():
    east = EarthLocation.from_geodetic(lon=180.0 * u.deg, lat=10.0 * u.deg, height=0.0 * u.m)
    west = EarthLocation.from_geodetic(lon=-180.0 * u.deg, lat=10.0 * u.deg, height=0.0 * u.m)
    _, v_east = station_gcrs_state(east, EPOCH)
    _, v_west = station_gcrs_state(west, EPOCH)
    difference = np.linalg.norm(v_east - v_west)
    assert difference < 1.0e-12, f"lon +180 and -180 differ by {difference:.3e} km/s"


def test_elevated_station_moves_faster():
    ground = EarthLocation.from_geodetic(lon=30.0 * u.deg, lat=40.0 * u.deg, height=0.0 * u.m)
    peak = EarthLocation.from_geodetic(lon=30.0 * u.deg, lat=40.0 * u.deg, height=10.0 * u.km)
    _, v_ground = station_gcrs_state(ground, EPOCH)
    _, v_peak = station_gcrs_state(peak, EPOCH)
    ratio = np.linalg.norm(v_peak) / np.linalg.norm(v_ground)
    expected_ratio = axis_distance_km(peak) / axis_distance_km(ground)
    assert abs(ratio - expected_ratio) < 1.0e-9, (
        f"speed ratio {ratio:.12f} vs distance-from-axis ratio {expected_ratio:.12f}"
    )


def test_geocentre_has_zero_velocity():
    centre = EarthLocation.from_geocentric(0.0, 0.0, 0.0, unit=u.m)
    r_site, v_site = station_gcrs_state(centre, EPOCH)
    assert np.all(np.isfinite(r_site)) and np.all(np.isfinite(v_site)), "geocentre gives non-finite state"
    assert np.linalg.norm(v_site) == 0.0, f"geocentre speed {np.linalg.norm(v_site):.3e} km/s"


def test_velocity_is_continuous_across_leap_second():
    # A leap second was inserted at 2012-06-30T23:59:60 UTC. Astropy's time
    # arithmetic steps in SI seconds, so the station's motion over the step
    # must stay a smooth rotation, not jump by a second's worth (~0.4 km).
    time = Time("2012-06-30T23:59:59.5", scale="utc")
    _, v_before = station_gcrs_state(ZIMMERWALD, time - 1.0 * u.s)
    _, v_after = station_gcrs_state(ZIMMERWALD, time + 1.0 * u.s)
    v_fd = finite_difference_velocity(ZIMMERWALD, time)
    v_mid = 0.5 * (v_before + v_after)
    error = relative_error(v_fd, v_mid)
    assert error < FD_REL_TOL, f"finite difference across leap second: rel err {error:.2e}"


def test_full_sidereal_day_returns_station_to_start():
    # Integrating the velocity over one rotation must bring the station back.
    # Precession over a day moves it ~4 m, so allow 50 m.
    sidereal_day_s = 2.0 * np.pi / OMEGA_EARTH
    r_start, _ = station_gcrs_state(ZIMMERWALD, EPOCH)
    r_end, _ = station_gcrs_state(ZIMMERWALD, EPOCH + sidereal_day_s * u.s)
    miss_km = np.linalg.norm(r_end - r_start)
    assert miss_km < 0.05, f"after one sidereal day the station is {miss_km * 1000:.1f} m away"

    r_half, _ = station_gcrs_state(ZIMMERWALD, EPOCH + 0.5 * sidereal_day_s * u.s)
    assert np.linalg.norm(r_half - r_start) > 2.0 * axis_distance_km(ZIMMERWALD) - 0.05, (
        "after half a sidereal day the station should be on the far side of the axis"
    )


# ----------------------------------------------------------------------------
# 5. Generated data
# ----------------------------------------------------------------------------

def test_generated_data_station_velocity():
    original = lab.paths.DATA_DIR
    with tempfile.TemporaryDirectory() as folder:
        lab.paths.DATA_DIR = Path(folder)
        try:
            generate_object()
            data = load_data("clean")
        finally:
            lab.paths.DATA_DIR = original

    assert data.shape[1] == NUM_OF_OUTPUT_COLUMNS, (
        f"data has {data.shape[1]} columns, expected {NUM_OF_OUTPUT_COLUMNS}"
    )

    r_site, v_site, time = data[:, 8:11], data[:, 11:14], data[:, 14]
    tracklet_id = data[:, 15].astype(int)

    radius = np.linalg.norm(r_site, axis=1)
    speed = np.linalg.norm(v_site, axis=1)
    radial = np.einsum("ij,ij->i", v_site, r_site) / radius

    # Expected speed from the row's own position: omega_E * distance from the
    # spin axis. Using the GCRS z axis here ignores the 0.17 deg pole tilt,
    # which changes rho by at most R * 0.003 -- hence the looser tolerance.
    rho = np.hypot(r_site[:, 0], r_site[:, 1])
    speed_error = np.abs(speed - OMEGA_EARTH * rho) / (OMEGA_EARTH * radius)

    assert np.all((6350.0 < radius) & (radius < 6400.0)), "station radius is not in km"
    assert np.max(np.abs(radial)) < 1.0e-9, f"max radial velocity {np.max(np.abs(radial)):.3e} km/s"
    assert np.max(speed_error) < 5.0e-3, f"max speed error {np.max(speed_error):.2e} of omega_E * R"

    # Within each tracklet, the middle row's velocity must match the central
    # difference of its neighbours' positions.
    worst = 0.0
    for tid in np.unique(tracklet_id):
        rows = np.flatnonzero(tracklet_id == tid)
        for k in range(1, len(rows) - 1):
            before, mid, after = rows[k - 1], rows[k], rows[k + 1]
            v_fd = (r_site[after] - r_site[before]) / (time[after] - time[before])
            worst = max(worst, relative_error(v_site[mid], v_fd))
    # Over 40 s the central difference has truncation (omega*20 s)^2 / 6 ~ 4e-7.
    assert worst < 1.0e-5, f"worst in-tracklet finite-difference rel err {worst:.2e}"


TESTS = [
    test_units_are_km_and_km_per_s,
    test_speed_matches_earth_rotation,
    test_velocity_is_horizontal,
    test_velocity_is_eastward_along_omega_cross_r,
    test_velocity_matches_finite_difference_of_position,
    test_velocity_is_consistent_across_the_generator_span,
    test_fix_station_to_earth_reproduces_the_chosen_point,
    test_fix_station_to_earth_gives_a_rotating_station,
    test_topocentric_rates_use_station_velocity_correctly,
    test_north_pole_station_is_almost_stationary,
    test_south_pole_station_is_almost_stationary,
    test_equator_station_has_maximum_speed,
    test_speed_follows_cos_latitude,
    test_southern_hemisphere_still_rotates_eastward,
    test_antimeridian_longitudes_agree,
    test_elevated_station_moves_faster,
    test_geocentre_has_zero_velocity,
    test_velocity_is_continuous_across_leap_second,
    test_full_sidereal_day_returns_station_to_start,
    test_generated_data_station_velocity,
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
