"""Validation suite for src/lab/data/gcrs_data_generation.py.

Run directly:

    uv run python test/test_gcrs_data.py

Every test generates into a temporary folder, so nothing here reads or writes
tracked or run data. The expected values come from the rows themselves or from
independent geometry, not from the generator's own intermediate results:

  1. Layout            Row counts, column count and IDs.
  2. Geometry          RA/Dec equal the direction of satellite minus station, and
                       stations are placed as recorded.
  3. Propagation       Every row follows from the object's recorded initial state.
  4. Population        Drawn orbits stay inside the recorded ranges.
  5. Timing            Tracklets follow the schedule, with offset and jitter.
  6. Noise             Chains differ only in angles, by bias plus scaled noise.
  7. Reproducibility   Object n is identical whichever run creates it.
  8. File format       Text files read back exactly, and settings reach the stages.
  9. Guards            Inconsistent data raises before anything is written.
"""

import contextlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import lab.paths
from lab.data.dataset import (
    NOISY_VARIANTS,
    VARIANTS,
    data_file,
    load_data,
    load_settings,
    population,
    settings_file,
    uncertainties_arcsec,
)
from lab.data.gcrs_data_generation import (
    NUM_OF_OUTPUT_COLUMNS,
    R_EARTH,
    generate_object,
    gcrs_to_radec,
    mu as MU,
    two_body,
)

ARCSEC = np.deg2rad(1.0 / 3600.0)

NUM_OBJECTS = 3
TRACKLETS_PER_OBJECT = 5
OBSERVATIONS_PER_TRACKLET = 5
ROWS_PER_OBJECT = TRACKLETS_PER_OBJECT * OBSERVATIONS_PER_TRACKLET

# Lowest elevation the 45 deg station limit allows, at the smallest orbit.
MIN_ELEVATION_DEG = 37.0

# Satellite, station and time columns: everything but the measured angles.
NON_ANGLE_COLUMNS = [c for c in range(NUM_OF_OUTPUT_COLUMNS) if c not in (6, 7)]


@contextlib.contextmanager
def dataset(num_objects=NUM_OBJECTS):
    """Generate objects into a temporary folder, and point lab.paths at it."""
    original = lab.paths.DATA_DIR
    with tempfile.TemporaryDirectory() as folder:
        lab.paths.DATA_DIR = Path(folder)
        try:
            for _ in range(num_objects):
                generate_object()
            yield Path(folder)
        finally:
            lab.paths.DATA_DIR = original


def load_record():
    with settings_file().open() as handle:
        return json.load(handle)


def wrapped_difference(a, b):
    """a - b for angles, in (-pi, pi]."""
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def expect_error(message_part):
    """Assert that the next generate_object() raises ValueError with this text."""
    try:
        generate_object()
    except ValueError as exc:
        assert message_part in str(exc), f"unexpected error: {exc}"
    else:
        raise AssertionError(f"no error raised, expected {message_part!r}")


def snapshot():
    """The content of every data file, to prove a guard wrote nothing."""
    paths = [data_file(variant) for variant in VARIANTS] + [settings_file()]
    return [path.read_bytes() for path in paths]


# ----------------------------------------------------------------------------
# 1. Layout
# ----------------------------------------------------------------------------

def test_rows_and_columns():
    with dataset():
        for variant in VARIANTS:
            data = load_data(variant)
            assert data.shape == (NUM_OBJECTS * ROWS_PER_OBJECT, NUM_OF_OUTPUT_COLUMNS), (
                f"{variant} has shape {data.shape}"
            )


def test_ids_are_contiguous():
    with dataset():
        data = load_data("clean")
    assert np.array_equal(data[:, 17], np.arange(len(data))), "observation IDs have gaps"
    assert np.array_equal(
        data[:, 15], np.repeat(np.arange(len(data) // 5), OBSERVATIONS_PER_TRACKLET),
    ), "tracklet IDs are not one per five rows"
    assert np.array_equal(
        data[:, 16], np.repeat(np.arange(NUM_OBJECTS), ROWS_PER_OBJECT),
    ), "object IDs are not one per twenty-five rows"


# ----------------------------------------------------------------------------
# 2. Geometry
# ----------------------------------------------------------------------------

def test_clean_angles_point_from_station_to_satellite():
    with dataset():
        data = load_data("clean")
    line_of_sight = data[:, :3] - data[:, 8:11]
    ra = np.arctan2(line_of_sight[:, 1], line_of_sight[:, 0]) % (2.0 * np.pi)
    dec = np.arcsin(line_of_sight[:, 2] / np.linalg.norm(line_of_sight, axis=1))
    assert np.max(np.abs(wrapped_difference(ra, data[:, 6]))) < 1.0e-13, "RA does not match"
    assert np.max(np.abs(dec - data[:, 7])) < 1.0e-13, "Dec does not match"


def test_recorded_station_offset_is_the_true_offset():
    with dataset():
        data = load_data("clean")
        objects = load_record()["objects"]
    for tracklet in (t for obj in objects for t in obj["tracklets"]):
        row = data[data[:, 15] == tracklet["tracklet_id"]][0]
        cosine = np.dot(row[:3], row[8:11]) / (
            np.linalg.norm(row[:3]) * np.linalg.norm(row[8:11])
        )
        actual = np.degrees(np.arccos(cosine))
        recorded = tracklet["station_offset_from_subsatellite_deg"]
        assert abs(actual - recorded) < 1.0e-6, f"recorded {recorded}, actual {actual}"
        assert recorded <= 45.0, f"offset {recorded} deg is above the limit"


def test_every_observation_is_high_in_the_sky():
    with dataset():
        data = load_data("clean")
    line_of_sight = data[:, :3] - data[:, 8:11]
    up = data[:, 8:11] / np.linalg.norm(data[:, 8:11], axis=1, keepdims=True)
    sine = np.sum(up * line_of_sight, axis=1) / np.linalg.norm(line_of_sight, axis=1)
    elevation = np.degrees(np.arcsin(sine))
    assert elevation.min() > MIN_ELEVATION_DEG, f"lowest elevation {elevation.min():.1f} deg"


def test_station_velocity_is_rigid_earth_rotation():
    with dataset():
        data = load_data("clean")
    r_site, v_site = data[:, 8:11], data[:, 11:14]
    radius = np.linalg.norm(r_site, axis=1)
    radial = np.sum(v_site * r_site, axis=1) / radius
    assert np.all(np.abs(radius - R_EARTH / 1000.0) < 25.0), "station is not at Earth's surface"
    assert np.max(np.abs(radial)) < 1.0e-9, f"radial speed {np.max(np.abs(radial)):.1e} km/s"


# ----------------------------------------------------------------------------
# 3. Propagation
# ----------------------------------------------------------------------------

def test_rows_follow_from_the_recorded_initial_state():
    with dataset():
        data = load_data("clean")
        objects = load_record()["objects"]
    for obj in objects:
        state = np.array(obj["initial_state_km_km_s"])
        for row in data[data[:, 16] == obj["object_id"]]:
            r, v = two_body(MU, row[14], state[:3], state[3:])
            assert np.linalg.norm(r - row[:3]) < 1.0e-6, "position does not propagate"
            assert np.linalg.norm(v - row[3:6]) < 1.0e-9, "velocity does not propagate"


def test_orbit_elements_stay_constant_along_each_object():
    with dataset():
        data = load_data("clean")
    for object_id in range(NUM_OBJECTS):
        rows = data[data[:, 16] == object_id]
        r, v = rows[:, :3], rows[:, 3:6]
        energy = np.sum(v * v, axis=1) / 2.0 - MU / np.linalg.norm(r, axis=1)
        momentum = np.cross(r, v)
        assert np.ptp(energy) / abs(energy[0]) < 1.0e-9, "energy is not conserved"
        assert np.max(np.linalg.norm(momentum - momentum[0], axis=1)) < 1.0e-5, (
            "angular momentum is not conserved"
        )


# ----------------------------------------------------------------------------
# 4. Population
# ----------------------------------------------------------------------------

def test_orbits_are_inside_the_recorded_population():
    with dataset(num_objects=8):
        settings = load_settings()
        data = load_data("clean")
    a_min, a_max = settings["semi_major_axis_range_km"]
    e_min, e_max = settings["eccentricity_range"]
    i_min, i_max = settings["inclination_range_deg"]

    for object_id in range(8):
        row = data[data[:, 16] == object_id][0]
        r, v = row[:3], row[3:6]
        r_mag = np.linalg.norm(r)
        h = np.cross(r, v)
        a = 1.0 / (2.0 / r_mag - np.dot(v, v) / MU)
        e = np.linalg.norm(np.cross(v, h) / MU - r / r_mag)
        i = np.degrees(np.arccos(h[2] / np.linalg.norm(h)))
        assert a_min <= a <= a_max, f"a = {a}"
        assert e_min <= e <= e_max, f"e = {e}"
        assert i_min <= i <= i_max, f"i = {i}"


# ----------------------------------------------------------------------------
# 5. Timing
# ----------------------------------------------------------------------------

def test_tracklets_follow_the_schedule():
    with dataset(num_objects=8):
        settings = load_settings()
        data = load_data("clean")
    schedule = np.array(settings["tracklet_schedule_h"])
    jitter = settings["max_tracklet_jitter_h"]

    first_starts = []
    for object_id in range(8):
        rows = data[data[:, 16] == object_id]
        starts = rows[::OBSERVATIONS_PER_TRACKLET, 14] / 3600.0
        first_starts.append(starts[0])
        assert 0.0 <= starts[0] <= settings["max_start_offset_h"], f"offset {starts[0]} h"
        assert np.all(np.abs(starts - starts[0] - schedule) <= jitter), (
            f"object {object_id} is off schedule: {starts - starts[0]}"
        )

    assert len(set(first_starts)) == 8, "objects share a start time"


def test_tracklet_duration_and_spacing():
    with dataset():
        data = load_data("clean")
    for tracklet_id in np.unique(data[:, 15]):
        times = data[data[:, 15] == tracklet_id, 14]
        assert np.allclose(np.diff(times), 20.0), f"spacing {np.diff(times)}"


def test_every_pair_has_a_time_gap():
    # A pair at one epoch cannot be tested, because Lambert needs two.
    with dataset():
        data = load_data("clean")
    starts = data[::OBSERVATIONS_PER_TRACKLET, 14]
    assert len(set(starts)) == len(starts), "two tracklets share a start time"


# ----------------------------------------------------------------------------
# 6. Noise
# ----------------------------------------------------------------------------

def test_chains_differ_only_in_angles():
    with dataset():
        clean = load_data("clean")
        for variant in NOISY_VARIANTS:
            noisy = load_data(variant)
            assert np.array_equal(noisy[:, NON_ANGLE_COLUMNS], clean[:, NON_ANGLE_COLUMNS]), (
                f"{variant} differs outside the angle columns"
            )
            assert not np.array_equal(noisy[:, 6:8], clean[:, 6:8]), f"{variant} has no noise"


def test_noise_is_bias_plus_scaled_shared_draws():
    with dataset():
        clean = load_data("clean")
        objects = load_record()["objects"]
        chains = {variant: load_data(variant) for variant in NOISY_VARIANTS}

    tracklets = [t for obj in objects for t in obj["tracklets"]]
    unit_draws = {}
    for variant, sigma in NOISY_VARIANTS.items():
        noise = np.column_stack((
            wrapped_difference(chains[variant][:, 6], clean[:, 6]),
            chains[variant][:, 7] - clean[:, 7],
        )) / ARCSEC
        for tracklet in tracklets:
            rows = clean[:, 15] == tracklet["tracklet_id"]
            noise[rows, 0] -= tracklet["ra_systematic_bias_arcsec"]
            noise[rows, 1] -= tracklet["dec_systematic_bias_arcsec"]
        unit_draws[variant] = noise / sigma

    reference = unit_draws["noisy_0.7as"]
    for variant, draws in unit_draws.items():
        assert np.max(np.abs(draws - reference)) < 1.0e-5, f"{variant} draws differ"
    assert 0.7 < reference.std() < 1.3, f"unit draws have sd {reference.std():.2f}"


def test_systematic_bias_has_the_recorded_size():
    with dataset(num_objects=8):
        settings = load_settings()
        tracklets = [t for obj in load_record()["objects"] for t in obj["tracklets"]]
    biases = np.array([
        value for t in tracklets
        for value in (t["ra_systematic_bias_arcsec"], t["dec_systematic_bias_arcsec"])
    ])
    expected = settings["systematic_noise_arcsec"]
    assert 0.7 * expected < biases.std() < 1.3 * expected, f"bias sd {biases.std():.2f}"


def test_bias_is_constant_within_a_tracklet():
    with dataset():
        clean = load_data("clean")
        noisy = load_data("noisy_0.7as")
    noise = (noisy[:, 7] - clean[:, 7]) / ARCSEC
    within = [noise[clean[:, 15] == t].std() for t in np.unique(clean[:, 15])]
    between = np.array([noise[clean[:, 15] == t].mean() for t in np.unique(clean[:, 15])])
    assert np.mean(within) < 2.0 * 0.7, f"scatter within a tracklet {np.mean(within):.2f}"
    assert between.std() > 2.0, f"tracklets share one bias, sd {between.std():.2f}"


# ----------------------------------------------------------------------------
# 7. Reproducibility
# ----------------------------------------------------------------------------

def test_generation_is_reproducible():
    with dataset():
        first = snapshot()
    with dataset():
        second = snapshot()
    assert first == second, "the same object count gave different files"


def test_object_n_does_not_depend_on_the_object_count():
    with dataset(num_objects=1):
        one = data_file("noisy_2.0as").read_bytes()
    with dataset(num_objects=3):
        three = data_file("noisy_2.0as").read_bytes()
    assert three.startswith(one), "object 0 changed when more objects were generated"


def test_objects_differ():
    with dataset():
        data = load_data("clean")
    first = data[data[:, 16] == 0, :6]
    second = data[data[:, 16] == 1, :6]
    assert np.linalg.norm(first - second) > 1.0, "two objects have the same orbit"


# ----------------------------------------------------------------------------
# 8. File format
# ----------------------------------------------------------------------------

def test_text_files_read_back_exactly():
    with dataset():
        path = data_file("noisy_0.7as")
        data = load_data("noisy_0.7as")
        text_rows = [line.split(",") for line in path.read_text().splitlines()]
    for values, row in zip(text_rows, data):
        for text, value in zip(values, row):
            assert float(text) == value, f"{text} read back as {value!r}"


def test_settings_reach_the_later_stages():
    with dataset():
        assert uncertainties_arcsec("clean") == (0.0, 0.0)
        assert uncertainties_arcsec("noisy_1.2as") == (1.2, 5.0)
        a_min, a_max, e_max = population()
    assert (a_min, a_max, e_max) == (38000.0, 45000.0, 0.01), "population does not match"


def test_gcrs_to_radec_is_consistent_for_all_quadrants():
    for x, y in ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)):
        ra, dec = gcrs_to_radec(np.array([x, y, 0.0]))
        assert 0.0 <= ra < 2.0 * np.pi, f"RA {ra} is out of range"
        assert abs(ra - np.arctan2(y, x) % (2.0 * np.pi)) < 1.0e-14
        assert abs(dec) < 1.0e-14


# ----------------------------------------------------------------------------
# 9. Guards
# ----------------------------------------------------------------------------

def test_out_of_step_files_are_refused():
    with dataset():
        path = data_file("noisy_1.2as")
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(lines[:-1]))
        before = snapshot()
        expect_error("out of sync")
        assert snapshot() == before, "the guard let data be written"


def test_changed_settings_are_refused():
    with dataset():
        record = load_record()
        record["settings"]["systematic_noise_arcsec"] = 1.0
        settings_file().write_text(json.dumps(record))
        before = snapshot()
        expect_error("differ")
        assert snapshot() == before, "the guard let data be written"


def test_missing_settings_record_is_refused():
    with dataset():
        settings_file().unlink()
        expect_error("missing")


def test_settings_record_behind_the_data_is_refused():
    # A run that crashed after writing rows but before recording its object.
    with dataset():
        record = load_record()
        record["objects"].pop()
        settings_file().write_text(json.dumps(record))
        before = snapshot()
        expect_error("records")
        assert snapshot() == before, "the guard let data be written"


def test_wrong_column_count_is_refused():
    with dataset():
        path = data_file("clean")
        path.write_text(path.read_text().rstrip("\n") + ",0\n")
        expect_error("columns")


def test_help_generates_nothing():
    script = Path(sys.modules["lab.data.gcrs_data_generation"].__file__)
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run(
            [sys.executable, str(script), "--help"],
            env={**os.environ, "LAB_DATA_DIR": folder},
            capture_output=True, check=True,
        )
        assert list(Path(folder).iterdir()) == [], "--help wrote files"


TESTS = [
    test_rows_and_columns,
    test_ids_are_contiguous,
    test_clean_angles_point_from_station_to_satellite,
    test_recorded_station_offset_is_the_true_offset,
    test_every_observation_is_high_in_the_sky,
    test_station_velocity_is_rigid_earth_rotation,
    test_rows_follow_from_the_recorded_initial_state,
    test_orbit_elements_stay_constant_along_each_object,
    test_orbits_are_inside_the_recorded_population,
    test_tracklets_follow_the_schedule,
    test_tracklet_duration_and_spacing,
    test_every_pair_has_a_time_gap,
    test_chains_differ_only_in_angles,
    test_noise_is_bias_plus_scaled_shared_draws,
    test_systematic_bias_has_the_recorded_size,
    test_bias_is_constant_within_a_tracklet,
    test_generation_is_reproducible,
    test_object_n_does_not_depend_on_the_object_count,
    test_objects_differ,
    test_text_files_read_back_exactly,
    test_settings_reach_the_later_stages,
    test_gcrs_to_radec_is_consistent_for_all_quadrants,
    test_out_of_step_files_are_refused,
    test_changed_settings_are_refused,
    test_missing_settings_record_is_refused,
    test_settings_record_behind_the_data_is_refused,
    test_wrong_column_count_is_refused,
    test_help_generates_nothing,
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
