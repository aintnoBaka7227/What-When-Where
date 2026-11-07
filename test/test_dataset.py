"""Tests for src/lab/data/dataset.py, how later stages read stage 1.

    uv run python test/test_dataset.py

Column layout against the generator, file reading, call-time paths and shared
constants. See TESTING.md.
"""

import tempfile
from pathlib import Path

import numpy as np

import lab.paths
from lab.data import dataset
from lab.data import gcrs_data_generation as generator
from pipeline_fixture import pipeline, run_tests


# --------------------------------------------------------------------------
# 1. layout
# --------------------------------------------------------------------------

def test_layout_matches_the_generator_constants():
    assert dataset.NUM_COLUMNS == generator.NUM_OF_OUTPUT_COLUMNS
    assert dataset.TRACKLET_ID_COLUMN == generator.TRACKLET_ID_COLUMN
    assert dataset.OBJECT_ID_COLUMN == generator.OBJECT_ID_COLUMN
    assert dataset.OBSERVATION_ID_COLUMN == generator.OBSERVATION_ID_COLUMN


def test_layout_matches_the_recorded_column_names():
    """The settings file names every column; each name must sit where we read it."""
    expected = {
        "rx_km": dataset.SATELLITE_POSITION.start,
        "vx_km_s": dataset.SATELLITE_VELOCITY.start,
        "ra_rad": dataset.RA_COLUMN,
        "dec_rad": dataset.DEC_COLUMN,
        "site_x_km": dataset.SITE_POSITION.start,
        "site_vx_km_s": dataset.SITE_VELOCITY.start,
        "time_s": dataset.TIME_COLUMN,
        "tracklet_id": dataset.TRACKLET_ID_COLUMN,
        "object_id": dataset.OBJECT_ID_COLUMN,
        "observation_id": dataset.OBSERVATION_ID_COLUMN,
    }
    with pipeline(num_objects=1, stages=1):
        columns = dataset.load_settings()["columns"]

    assert len(columns) == dataset.NUM_COLUMNS
    for name, index in expected.items():
        assert columns[index] == name, f"column {index} is {columns[index]}, expected {name}"

    # The slices cover x, y, z in order.
    for vector in (dataset.SATELLITE_POSITION, dataset.SATELLITE_VELOCITY,
                   dataset.SITE_POSITION, dataset.SITE_VELOCITY):
        assert vector.stop - vector.start == 3


# --------------------------------------------------------------------------
# 2. reading
# --------------------------------------------------------------------------

def test_load_data_rejects_an_old_layout():
    """A 17-column file from an older generator must fail loudly, not misread."""
    original = lab.paths.DATA_DIR
    with tempfile.TemporaryDirectory() as folder:
        lab.paths.DATA_DIR = Path(folder)
        try:
            np.savetxt(dataset.data_file("clean"), np.zeros((3, 17)), delimiter=",")
            try:
                dataset.load_data("clean")
            except ValueError as exc:
                assert "17 columns" in str(exc)
            else:
                raise AssertionError("a 17-column file was accepted")
        finally:
            lab.paths.DATA_DIR = original


def test_load_data_keeps_a_single_row_two_dimensional():
    """Edge case: one row must still index as data[:, column]."""
    original = lab.paths.DATA_DIR
    with tempfile.TemporaryDirectory() as folder:
        lab.paths.DATA_DIR = Path(folder)
        try:
            np.savetxt(dataset.data_file("clean"), np.zeros((1, 18)), delimiter=",")
            assert dataset.load_data("clean").shape == (1, 18)
        finally:
            lab.paths.DATA_DIR = original


def _rows(times, object_id=0, tracklet_id=0):
    rows = np.zeros((len(times), dataset.NUM_COLUMNS))
    rows[:, dataset.TIME_COLUMN] = times
    rows[:, dataset.OBJECT_ID_COLUMN] = object_id
    rows[:, dataset.TRACKLET_ID_COLUMN] = tracklet_id
    return rows


def test_tracklet_rows_sorts_by_time():
    data = np.vstack([_rows([40.0, 0.0, 20.0]), _rows([5.0], tracklet_id=1)])
    rows = dataset.tracklet_rows(data, 0, 0)
    assert rows[:, dataset.TIME_COLUMN].tolist() == [0.0, 20.0, 40.0]


def test_tracklet_rows_separates_objects_sharing_a_tracklet_id():
    """Edge case: selection uses both IDs, not the tracklet ID alone."""
    data = np.vstack([_rows([0.0, 20.0], object_id=0), _rows([7.0], object_id=1)])
    assert len(dataset.tracklet_rows(data, 0, 0)) == 2
    assert len(dataset.tracklet_rows(data, 1, 0)) == 1


def test_tracklet_rows_raises_for_an_unknown_tracklet():
    try:
        dataset.tracklet_rows(_rows([0.0]), 3, 9)
    except KeyError:
        return
    raise AssertionError("an unknown tracklet returned rows")


def test_tracklet_ids_are_unique_and_sorted():
    data = np.vstack([_rows([0.0, 1.0], 1, 5), _rows([0.0], 0, 2), _rows([0.0], 1, 5)])
    assert dataset.tracklet_ids(data) == [(0, 2), (1, 5)]


# --------------------------------------------------------------------------
# 3. paths
# --------------------------------------------------------------------------

def test_stage_paths_follow_data_dir_after_import():
    """Reassigning DATA_DIR after import must move every stage's files, as run_model.py does."""
    from lab.observation.create_attributes_vector import ATTRIBUTABLES_NPZ_FILES
    from lab.hypothesis.range_bounds import ATTRIBUTABLES_FILES, PRECISION_OUTPUT_FILES
    from lab.hypothesis.k_bounds import PRECISION_OUTPUT_FILES as K_FILES
    from lab.model.build_orbit import RANGE_BOUNDS_FILES

    original = lab.paths.DATA_DIR
    try:
        lab.paths.DATA_DIR = Path("/somewhere/else")
        for mapping in (ATTRIBUTABLES_NPZ_FILES, ATTRIBUTABLES_FILES,
                        PRECISION_OUTPUT_FILES, K_FILES, RANGE_BOUNDS_FILES):
            for variant in dataset.VARIANTS:
                assert mapping[variant].parent == Path("/somewhere/else"), mapping[variant]
    finally:
        lab.paths.DATA_DIR = original


def test_stage_paths_cover_exactly_the_chains():
    from lab.hypothesis.range_bounds import ATTRIBUTABLES_FILES
    from lab.observation.create_attributes_vector import ATTRIBUTABLES_NPZ_FILES

    assert tuple(ATTRIBUTABLES_FILES) == dataset.VARIANTS
    # Stage 2's own copy and stage 3's registry must name the same files.
    for variant in dataset.VARIANTS:
        assert ATTRIBUTABLES_FILES[variant] == ATTRIBUTABLES_NPZ_FILES[variant]
    try:
        ATTRIBUTABLES_FILES["noisy_9.9as"]
    except KeyError:
        return
    raise AssertionError("an unknown chain got a path")


# --------------------------------------------------------------------------
# 4. constants
# --------------------------------------------------------------------------

def test_every_module_uses_the_generator_mu():
    from lab.hypothesis import k_bounds
    from lab.model import build_orbit
    from lab.scoring import discriminator_loss
    from lab.search import optimise_hypothesis
    from lab.evaluation import metrics

    for module in (k_bounds, build_orbit, discriminator_loss, optimise_hypothesis, metrics):
        assert module.MU == generator.mu, f"{module.__name__}.MU = {module.MU}"


def test_population_follows_the_generator_test_case():
    """Stages 3, 4 and 8 read the orbit family from the settings, not a constant."""
    for test_case, e_max in (("a", 0.01), ("b", 0.2)):
        with pipeline(num_objects=1, stages=1, test_case=test_case):
            assert dataset.population() == (38000.0, 45000.0, e_max), test_case


TESTS = [
    test_layout_matches_the_generator_constants,
    test_layout_matches_the_recorded_column_names,
    test_load_data_rejects_an_old_layout,
    test_load_data_keeps_a_single_row_two_dimensional,
    test_tracklet_rows_sorts_by_time,
    test_tracklet_rows_separates_objects_sharing_a_tracklet_id,
    test_tracklet_rows_raises_for_an_unknown_tracklet,
    test_tracklet_ids_are_unique_and_sorted,
    test_stage_paths_follow_data_dir_after_import,
    test_stage_paths_cover_exactly_the_chains,
    test_every_module_uses_the_generator_mu,
    test_population_follows_the_generator_test_case,
]


if __name__ == "__main__":
    raise SystemExit(run_tests(TESTS))
