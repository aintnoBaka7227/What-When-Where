"""Run the whole tracklet-association pipeline in order.

    uv run python main.py                  # run the pipeline on the existing dataset
    uv run python main.py --objects 5      # generate 5 new objects first, then run
    uv run python main.py --fresh -n 5     # delete the dataset, generate 5, then run
    uv run python main.py --tests          # run the test suites first, then the pipeline
    uv run python main.py --validate       # after the pipeline, trace every optimisation
    uv run python main.py --list           # show the stages and exit

Every stage runs as a separate process, the way you would run it by hand.
That keeps failures isolated -- one stage blowing up does not take the
interpreter with it -- and means this file needs no knowledge of each
stage's internals. Two of them (new_data_gen, optimise_hypothesis) do all
their work in a __main__ block and have no callable entry point, so
importing them was never an option anyway.

Output is echoed live and captured to runs/<timestamp>.log, so a run can be
diffed against an earlier one. That matters here: the association result is
a handful of numbers at the very end of a three-minute run, and a change
upstream moves them quietly. With --validate, a companion
runs/<timestamp>-validate.log (same timestamp) holds the optimiser trace,
and the per-evaluation CSVs are stamped to match.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# The stages that actually do something, in dependency order.
#
# modelled/model_rates.py and discriminator/discriminator_loss.py are
# deliberately absent. Both are pure libraries now -- they have no __main__
# and running them produces nothing. They are exercised by the stages that
# import them, and by their own test suites.
STAGES = [
    ("generate",     "simulation/new_data_gen.py",
     "draw one near-GEO object and append it to the clean and noisy datasets"),
    ("attributables", "observation/create_attributes_vector.py",
     "compress each tracklet's RA/Dec series into one attributable"),
    ("range bounds", "hypothesis/range_bounds.py",
     "bound the topocentric range per tracklet (Eqs. 23-24, 26-28)"),
    ("k bounds",     "hypothesis/k_bounds.py",
     "bound the revolution parameter per tracklet pair (Eq. 29)"),
    ("hypotheses",   "orbit/build_orbit.py",
     "solve Lambert for each range hypothesis (Eq. 5)"),
    ("associate",    "optimisation/optimise_hypothesis.py",
     "minimise the discriminator loss over every pair (Eq. 30)"),
]

# Run before the pipeline with --tests. Neither needs generated data.
TESTS = [
    ("lambert",       "orbit/test_lambert.py"),
    ("discriminator", "discriminator/test_discriminator_loss.py"),
]

DATASET_FILES = [
    PROJECT_ROOT / "simulation" / f"generated_data_{variant}.{suffix}"
    for variant in ("clean", "noisy")
    for suffix in ("npz", "txt")
]


class Tee:
    """Write to the terminal and the run log at once."""

    def __init__(self, log_file):
        self.log_file = log_file

    def write(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()
        if self.log_file is not None:
            self.log_file.write(text)
            self.log_file.flush()


def run_script(tee, relative_path, indent="      ", script_args=()):
    """Run one script, streaming its output. Returns (ok, seconds)."""
    started = time.monotonic()

    process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / relative_path), *script_args],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        if tee is not None:            # tee=None: drain the pipe silently
            tee.write(f"{indent}{line}")

    process.wait()
    return process.returncode == 0, time.monotonic() - started


def dataset_summary():
    """Objects and rows currently in the clean dataset, if there is one."""
    archive_path = PROJECT_ROOT / "simulation" / "generated_data_clean.npz"
    if not archive_path.exists():
        return "no dataset yet"

    import numpy as np  # local: keeps --list and --help fast

    with np.load(archive_path) as archive:
        data = archive["data"]
        test_case = str(archive["test_case"]) if "test_case" in archive else "?"

    objects = int(data[:, 16].max()) + 1
    tracklets = len(set(zip(data[:, 16].astype(int), data[:, 15].astype(int))))
    pairs = tracklets * (tracklets - 1) // 2

    return (
        f"test case {test_case}, {objects} objects, {tracklets} tracklets, "
        f"up to {pairs} pairs, {data.shape[0]} rows"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the tracklet-association pipeline end to end.",
    )
    parser.add_argument(
        "-n", "--objects", type=int, default=0,
        help="generate this many new objects before running (default: 0, "
             "use the existing dataset)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="delete the existing dataset first; requires --objects",
    )
    parser.add_argument(
        "--tests", action="store_true",
        help="run the test suites before the pipeline, and stop if they fail",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="after the pipeline, trace every optimisation into a companion "
             "runs/<timestamp>-validate.log (see optimisation/validate_optimiser.py)",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="do not write runs/<timestamp>.log",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="list the stages and exit",
    )
    args = parser.parse_args()

    if args.list:
        for index, (name, path, description) in enumerate(STAGES, 1):
            print(f"  {index}. {name:14s} {path}")
            print(f"     {description}")
        return 0

    if args.fresh and args.objects < 1:
        parser.error("--fresh deletes the dataset, so it needs --objects to rebuild it")

    stamp = f"{datetime.now():%Y-%m-%d-%H%M%S}"

    log_file = None
    log_path = None
    if not args.no_log:
        runs_directory = PROJECT_ROOT / "runs"
        runs_directory.mkdir(exist_ok=True)
        log_path = runs_directory / f"{stamp}.log"
        log_file = log_path.open("w")

    tee = Tee(log_file)
    rule = "=" * 68

    tee.write(f"{rule}\n")
    tee.write(f" What When Where -- tracklet association pipeline\n")
    tee.write(f" {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    tee.write(f"{rule}\n")

    if args.fresh:
        removed = 0
        for path in DATASET_FILES:
            if path.exists():
                path.unlink()
                removed += 1
        tee.write(f"\n  --fresh: removed {removed} dataset file(s)\n")

    if args.tests:
        tee.write(f"\n  running test suites first\n")
        for name, path in TESTS:
            tee.write(f"\n  [test] {name:14s} {path}\n")
            ok, seconds = run_script(tee, path)
            if not ok:
                tee.write(f"\n  FAILED: {path} -- stopping before the pipeline\n")
                return 1
            tee.write(f"      passed in {seconds:.1f}s\n")

    # The generate stage appends exactly one object per invocation, so N
    # objects means N runs. Everything after it is idempotent.
    plan = []
    for name, path, description in STAGES:
        if name == "generate":
            plan.extend([(name, path, description)] * args.objects)
        else:
            plan.append((name, path, description))

    if args.objects == 0:
        tee.write(f"\n  dataset: {dataset_summary()}\n")
        tee.write("  (pass --objects N to append more before running)\n")

    results = []
    total_started = time.monotonic()

    for index, (name, path, description) in enumerate(plan, 1):
        tee.write(f"\n[{index}/{len(plan)}] {name} -- {path}\n")
        tee.write(f"        {description}\n")

        ok, seconds = run_script(tee, path)
        results.append((name, path, ok, seconds))

        if not ok:
            tee.write(f"\n  FAILED: {path} exited non-zero. Stopping.\n")
            break

        tee.write(f"      done in {seconds:.1f}s\n")

        if name == "generate" and index == args.objects:
            tee.write(f"\n  dataset: {dataset_summary()}\n")

    total = time.monotonic() - total_started

    tee.write(f"\n{rule}\n")
    tee.write(" SUMMARY\n")
    for index, (name, path, ok, seconds) in enumerate(results, 1):
        tee.write(f"   {index:2d}. {name:14s} {'ok' if ok else 'FAILED':>7}  {seconds:7.1f}s\n")
    tee.write(f"       {'total':14s} {'':>7}  {total:7.1f}s\n")

    failed = [name for name, _, ok, _ in results if not ok]
    if failed:
        tee.write(f"\n  {len(failed)} stage(s) failed: {', '.join(failed)}\n")

    # Optimiser validation runs after the pipeline (it reads the derived
    # bounds files the pipeline just wrote). It writes its own
    # runs/<stamp>-validate.log -- given the same stamp as this run -- so its
    # ~1700 lines stay out of both the terminal and the main log.
    validate_path = None
    if args.validate and not failed:
        script_args = ["--stamp", stamp]
        if args.no_log:
            script_args.append("--no-log")
        else:
            validate_path = PROJECT_ROOT / "runs" / f"{stamp}-validate.log"

        tee.write("\n  --validate: tracing every optimisation\n")
        ok, seconds = run_script(
            None, "optimisation/validate_optimiser.py",
            script_args=tuple(script_args),
        )
        tee.write(f"      {'ok' if ok else 'FAILED':>7}  {seconds:7.1f}s\n")

    if log_path is not None:
        tee.write(f"\n  log: {log_path.relative_to(PROJECT_ROOT)}\n")
    if validate_path is not None:
        tee.write(f"  validate log: {validate_path.relative_to(PROJECT_ROOT)}\n")
    tee.write(f"{rule}\n")

    if log_file is not None:
        log_file.close()

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
