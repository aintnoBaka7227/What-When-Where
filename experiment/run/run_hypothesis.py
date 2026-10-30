"""Stages [3] and [4]: bound the range per tracklet and k per pair.

    uv run python experiment/run/run_hypothesis.py --parent <observation hash>
    uv run python experiment/run/run_hypothesis.py -n 4
"""

import run_observation
from lab.runner import execute, run_script, stage_parser


def run(parent, force=False):
    def body(tee, output_dir, config):
        return (
            run_script(tee, "src/lab/hypothesis/range_bounds.py", output_dir)
            and run_script(tee, "src/lab/hypothesis/k_bounds.py", output_dir)
        )

    return execute("hypothesis", {}, (parent, "observation"), body, force)


def build(objects):
    return run(run_observation.build(objects))


if __name__ == "__main__":
    args = stage_parser(__doc__, "observation").parse_args()
    run(args.parent or run_observation.build(args.objects), args.force)
