"""Stage [1]: generate a dataset. Has no inputs, so it always runs on its own.

    uv run python experiment/run/run_data.py -n 4
"""

import argparse
import json

import numpy as np

from lab.runner import execute, run_script


def dataset_summary(output_dir):
    """What the generator drew, from its settings record and the clean rows."""
    settings = json.loads(
        (output_dir / "generated_data_gcrs_settings.json").read_text()
    )["settings"]
    data = np.loadtxt(output_dir / "generated_data_gcrs_clean.txt", delimiter=",", ndmin=2)

    summary = {
        key: settings[key]
        for key in (
            "test_case", "semi_major_axis_range_km", "eccentricity_range",
            "inclination_range_deg", "relative_noise_levels_arcsec",
            "systematic_noise_arcsec",
        )
    }
    summary["objects"] = int(data[:, 16].max()) + 1
    summary["tracklets"] = len(
        set(zip(data[:, 16].astype(int), data[:, 15].astype(int)))
    )
    summary["rows"] = int(data.shape[0])
    return summary


def run(objects, force=False):
    def body(tee, output_dir, config):
        # The generator appends exactly one object per invocation, so N
        # objects means N runs.
        for _ in range(objects):
            if not run_script(tee, "src/lab/data/gcrs_data_generation.py", output_dir):
                return False
        config["dataset"] = dataset_summary(output_dir)
        return True

    return execute("data", {"objects": objects}, None, body, force)


def build(objects):
    return run(objects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--objects", type=int, required=True,
                        help="number of objects to generate")
    parser.add_argument("--force", action="store_true",
                        help="redo this stage if it already completed")
    args = parser.parse_args()
    run(args.objects, args.force)
