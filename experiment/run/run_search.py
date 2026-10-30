"""Stage [8]: find the best hypothesis for every tracklet pair.

This is the association test. Saves association_<variant>.npz.

    uv run python experiment/run/run_search.py --parent <hypothesis hash>
    uv run python experiment/run/run_search.py -n 4
"""

import numpy as np

import run_hypothesis
from lab.data.dataset import VARIANTS
from lab.evaluation.metrics import search_metrics
from lab.runner import execute, run_script, stage_parser, write_metrics


def association_summary(output_dir, variant):
    """True and false pair counts and loss ranges for one dataset chain."""
    with np.load(output_dir / f"association_{variant}.npz") as archive:
        is_true, loss = archive["is_true"], archive["loss"]

    summary = {}
    for name, losses in (("true_pairs", loss[is_true]), ("false_pairs", loss[~is_true])):
        summary[name] = {
            "n": int(losses.size),
            "min_loss": float(losses.min()) if losses.size else None,
            "max_loss": float(losses.max()) if losses.size else None,
        }
    return summary


def run(parent, force=False):
    def body(tee, output_dir, config):
        if not run_script(tee, "src/lab/search/optimise_hypothesis.py", output_dir):
            return False
        config["association"] = {
            variant: association_summary(output_dir, variant)
            for variant in VARIANTS
        }
        write_metrics(output_dir, search_metrics(output_dir))
        return True

    return execute("search", {}, (parent, "hypothesis"), body, force)


def build(objects):
    return run(run_hypothesis.build(objects))


if __name__ == "__main__":
    args = stage_parser(__doc__, "hypothesis").parse_args()
    run(args.parent or run_hypothesis.build(args.objects), args.force)
