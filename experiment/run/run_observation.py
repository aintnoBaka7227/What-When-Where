"""Stage [2]: compress each tracklet into an attributable.

    uv run python experiment/run/run_observation.py --parent <data hash>
    uv run python experiment/run/run_observation.py -n 4
"""

import run_data
from lab.runner import execute, run_script, stage_parser


def run(parent, force=False):
    def body(tee, output_dir, config):
        return run_script(tee, "src/lab/observation/create_attributes_vector.py", output_dir)

    return execute("observation", {}, (parent, "data"), body, force)


def build(objects):
    return run(run_data.build(objects))


if __name__ == "__main__":
    args = stage_parser(__doc__, "data").parse_args()
    run(args.parent or run_data.build(args.objects), args.force)
