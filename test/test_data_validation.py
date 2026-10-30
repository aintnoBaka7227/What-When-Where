"""Independent cross-checks of the generated data.

Run directly:

    uv run python test/test_data_validation.py

test/test_gcrs_data.py checks that the generator is self-consistent, and some of
its checks use the generator's own functions. These recompute the same quantities
by another route, so a mistake shared by the generator and its tests still shows:

  1. Propagation    two_body against a numerical integration of Newton's law.
  2. Earth rotation Station positions, through ERFA's IAU 2006/2000A rotation
                    called directly, against the recorded latitude and longitude.
  3. Angles         RA/Dec against astropy's own spherical conversion.

Every test generates into a temporary folder.
"""

import contextlib
import json
import tempfile
from pathlib import Path

import astropy.units as u
import erfa
import numpy as np
from astropy.coordinates import CartesianRepresentation, UnitSphericalRepresentation
from astropy.time import Time
from scipy.integrate import solve_ivp

import lab.paths
from lab.data.dataset import load_data, settings_file
from lab.data.gcrs_data_generation import generate_object, mu as MU, two_body

ARCSEC = np.deg2rad(1.0 / 3600.0)

# The integrator is the reference, so it runs far tighter than the comparison.
INTEGRATOR_RTOL = 1.0e-12
INTEGRATOR_ATOL = 1.0e-9   # km, km/s
PROPAGATION_TOL_KM = 1.0e-3  # 1 m after up to about 80 h

# Polar motion moves a site up to about 15 m. The direct rotation below leaves
# it out, so the recorded coordinates are compared to within that.
STATION_TOL_M = 30.0

# A fixed station stays put in the Earth-fixed frame; this covers the rounding.
EARTH_FIXED_TOL_M = 0.01


@contextlib.contextmanager
def dataset(num_objects=2):
    """Generate objects into a temporary folder, and point lab.paths at it."""
    original = lab.paths.DATA_DIR
    with tempfile.TemporaryDirectory() as folder:
        lab.paths.DATA_DIR = Path(folder)
        try:
            for _ in range(num_objects):
                generate_object()
            yield
        finally:
            lab.paths.DATA_DIR = original


def load_objects():
    with settings_file().open() as handle:
        return json.load(handle)["objects"]


def newton(_t, state):
    """Point-mass gravity, written out directly."""
    r = state[:3]
    return np.concatenate((state[3:], -MU * r / np.linalg.norm(r) ** 3))


def earth_fixed_km(r_gcrs_km, time):
    """Rotate a GCRS position into the Earth-fixed frame with ERFA directly."""
    matrix = erfa.c2t06a(
        time.tt.jd1, time.tt.jd2, time.ut1.jd1, time.ut1.jd2, 0.0, 0.0,
    )
    return matrix @ r_gcrs_km


def test_two_body_matches_numerical_integration():
    with dataset():
        data = load_data("clean")
        objects = load_objects()

    for obj in objects:
        rows = data[data[:, 16] == obj["object_id"]]
        state = np.array(obj["initial_state_km_km_s"])
        times = rows[:, 14]
        solution = solve_ivp(
            newton, (0.0, times[-1]), state, method="DOP853", t_eval=times,
            rtol=INTEGRATOR_RTOL, atol=INTEGRATOR_ATOL,
        )
        assert solution.success, solution.message

        position_error = np.linalg.norm(solution.y[:3].T - rows[:, :3], axis=1)
        assert position_error.max() < PROPAGATION_TOL_KM, (
            f"object {obj['object_id']}: {position_error.max():.2e} km from the integration"
        )


def test_two_body_returns_after_one_period():
    # A closed orbit: one period must bring the state back.
    with dataset(num_objects=1):
        obj = load_objects()[0]
    state = np.array(obj["initial_state_km_km_s"])
    period = obj["orbital_elements"]["period_h"] * 3600.0
    r, v = two_body(MU, period, state[:3], state[3:])
    assert np.linalg.norm(r - state[:3]) < 1.0e-4, "position did not return"
    assert np.linalg.norm(v - state[3:]) < 1.0e-7, "velocity did not return"


def test_stations_match_their_recorded_coordinates():
    with dataset():
        data = load_data("clean")
        objects = load_objects()
        epoch = Time(_settings_epoch(), scale="utc")

    for obj in objects:
        for tracklet in obj["tracklets"]:
            row = data[data[:, 15] == tracklet["tracklet_id"]][0]
            fixed_km = earth_fixed_km(row[8:11], epoch + row[14] * u.s)
            lon, lat, _height = erfa.gc2gd(erfa.WGS84, fixed_km * 1000.0)

            east = np.radians(tracklet["station_lon_deg"]) - lon
            east = (east + np.pi) % (2.0 * np.pi) - np.pi
            north = np.radians(tracklet["station_lat_deg"]) - lat
            miss_m = 6378137.0 * np.hypot(east * np.cos(lat), north)
            assert miss_m < STATION_TOL_M, (
                f"tracklet {tracklet['tracklet_id']} is {miss_m:.1f} m from its record"
            )


def test_stations_stay_fixed_to_the_ground():
    with dataset():
        data = load_data("clean")
        epoch = Time(_settings_epoch(), scale="utc")

    for tracklet_id in np.unique(data[:, 15]):
        rows = data[data[:, 15] == tracklet_id]
        fixed = np.array([
            earth_fixed_km(row[8:11], epoch + row[14] * u.s) for row in rows
        ])
        spread_m = np.max(np.linalg.norm(fixed - fixed[0], axis=1)) * 1000.0
        assert spread_m < EARTH_FIXED_TOL_M, (
            f"tracklet {tracklet_id} moved {spread_m:.3f} m on the ground"
        )


def test_angles_match_astropy_conversion():
    with dataset():
        data = load_data("clean")

    line_of_sight = data[:, :3] - data[:, 8:11]
    spherical = CartesianRepresentation(line_of_sight.T * u.km).represent_as(
        UnitSphericalRepresentation
    )
    ra = spherical.lon.to_value(u.rad)
    dec = spherical.lat.to_value(u.rad)
    ra_error = (ra - data[:, 6] + np.pi) % (2.0 * np.pi) - np.pi
    assert np.max(np.abs(ra_error)) / ARCSEC < 1.0e-8, "RA differs from astropy"
    assert np.max(np.abs(dec - data[:, 7])) / ARCSEC < 1.0e-8, "Dec differs from astropy"


def _settings_epoch():
    with settings_file().open() as handle:
        return json.load(handle)["settings"]["epoch_utc"]


TESTS = [
    test_two_body_matches_numerical_integration,
    test_two_body_returns_after_one_period,
    test_stations_match_their_recorded_coordinates,
    test_stations_stay_fixed_to_the_ground,
    test_angles_match_astropy_conversion,
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
