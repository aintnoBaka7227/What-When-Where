"""A real dataset from the current generator, for the stage 2-8 suites.

    with pipeline(num_objects=4) as run: ...

Generates into a temporary folder, points lab.paths.DATA_DIR there and runs
stages 2-4 in-process. Truth comes from the clean rows; only tests read it.
"""

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np

import lab.paths
from lab.data.dataset import (
    SATELLITE_POSITION,
    SATELLITE_VELOCITY,
    SITE_POSITION,
    SITE_VELOCITY,
    TIME_COLUMN,
    VARIANTS,
    load_data,
    tracklet_rows,
)
from lab.data.gcrs_data_generation import generate_object


class Run:
    """The files of one generated pipeline, with truth lookups."""

    def __init__(self, folder, stages):
        self.folder = folder
        self.stages = stages
        self.clean_rows = load_data("clean")

    def attributables(self, variant):
        from lab.observation.create_attributes_vector import ATTRIBUTABLES_NPZ_FILES
        return dict(np.load(ATTRIBUTABLES_NPZ_FILES[variant]))

    def range_bounds(self, variant):
        from lab.hypothesis.range_bounds import PRECISION_OUTPUT_FILES
        return dict(np.load(PRECISION_OUTPUT_FILES[variant]))

    def k_bounds(self, variant):
        from lab.hypothesis.k_bounds import PRECISION_OUTPUT_FILES
        return dict(np.load(PRECISION_OUTPUT_FILES[variant]))

    def truth_at_epoch(self, object_id, tracklet_id):
        """The clean row at a tracklet's mean time, which is its middle sample.

        Returns (t0, r_sat, v_sat, r_site, v_site).
        """
        rows = tracklet_rows(self.clean_rows, object_id, tracklet_id)
        t0 = rows[:, TIME_COLUMN].mean()
        middle = rows[np.argmin(np.abs(rows[:, TIME_COLUMN] - t0))]
        assert np.isclose(middle[TIME_COLUMN], t0, rtol=0.0, atol=1e-6), (
            "generator no longer samples tracklets evenly; t0 is not a sample"
        )
        return (
            float(middle[TIME_COLUMN]),
            middle[SATELLITE_POSITION].copy(),
            middle[SATELLITE_VELOCITY].copy(),
            middle[SITE_POSITION].copy(),
            middle[SITE_VELOCITY].copy(),
        )


def _quietly(function, *args):
    """Call a stage, discarding what it prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args)


@contextlib.contextmanager
def pipeline(num_objects=4, stages=4, test_case="a"):
    """Generate num_objects, then run stages 2 up to `stages` (1-4) on every chain."""
    from lab.observation.create_attributes_vector import (
        ATTRIBUTABLES_NPZ_FILES,
        ATTRIBUTABLES_TXT_FILES,
        build_attributables,
    )
    from lab.hypothesis.range_bounds import build_range_bounds
    from lab.hypothesis.k_bounds import build_k_bounds

    original = lab.paths.DATA_DIR
    with tempfile.TemporaryDirectory() as folder:
        lab.paths.DATA_DIR = Path(folder)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(num_objects):
                    generate_object(test_case)

            for variant in VARIANTS:
                if stages >= 2:
                    _quietly(
                        build_attributables, variant,
                        ATTRIBUTABLES_NPZ_FILES[variant],
                        ATTRIBUTABLES_TXT_FILES[variant],
                    )
                if stages >= 3:
                    _quietly(build_range_bounds, variant)
                if stages >= 4:
                    _quietly(build_k_bounds, variant)

            yield Run(Path(folder), stages)
        finally:
            lab.paths.DATA_DIR = original


def run_tests(tests):
    """The runner every suite in test/ shares: print each result, exit 1 on failure."""
    passed = 0
    failed = []

    for test in tests:
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
    print(f"{passed}/{len(tests)} passed")
    return 1 if failed else 0
