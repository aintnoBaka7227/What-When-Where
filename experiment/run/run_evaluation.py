"""Trace every stage-8 optimisation and report convergence.

The report goes to run.log only, because it is long. Saves
optimiser_history_<variant>.csv.

    uv run python experiment/run/run_evaluation.py --parent <search hash>
    uv run python experiment/run/run_evaluation.py -n 4
"""

import run_search
from lab.runner import Tee, execute, run_script, stage_parser


def run(parent, force=False):
    def body(tee, output_dir, config):
        return run_script(
            Tee(tee.log_file, quiet=True),
            "src/lab/evaluation/validate_optimiser.py", output_dir,
        )

    return execute("evaluation", {}, (parent, "search"), body, force)


if __name__ == "__main__":
    args = stage_parser(__doc__, "search").parse_args()
    run(args.parent or run_search.build(args.objects), args.force)
