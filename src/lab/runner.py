"""Run folders: what every experiment/run/run_<stage>.py script is built on.

Each run of a stage gets one folder, named by the hash of its config:

    results/runs/<hash>/config.json   what was run, and its status
    results/runs/<hash>/run.log       the terminal output, verbatim
    results/runs/<hash>/metrics.json  how the stage scored, if it has metrics
    results/runs/<hash>/output/       the stage's files, plus links to its inputs

The hash covers the stage, its parameters, its parent run and the code, so the
same config on the same code maps to the same folder. Generation is seeded and
the optimiser is deterministic, so a completed run is not repeated.

A stage reads and writes one folder, so a run's output folder starts with links
to everything its parent can see. config.json lists the files the stage itself
produced under "outputs".

Pipeline stages run as separate processes, the way you would run them by hand.
That keeps failures isolated -- one stage blowing up does not take the
interpreter with it. LAB_DATA_DIR points each one at its run's output folder.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from lab.paths import PROJECT_ROOT, RUNS_DIR


class Tee:
    """Write to the run log, and to the terminal unless quiet."""

    def __init__(self, log_file, quiet=False):
        self.log_file = log_file
        self.quiet = quiet

    def write(self, text):
        if not self.quiet:
            sys.stdout.write(text)
            sys.stdout.flush()
        self.log_file.write(text)
        self.log_file.flush()


def run_script(tee, relative_path, output_dir, indent="      "):
    """Run one stage script against a run's output folder. Returns True on success."""
    started = time.monotonic()
    tee.write(f"\n{relative_path}\n")

    process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / relative_path)],
        cwd=PROJECT_ROOT,
        env={**os.environ, "LAB_DATA_DIR": str(output_dir)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        tee.write(f"{indent}{line}")

    process.wait()
    ok = process.returncode == 0
    tee.write(f"{indent}{'done' if ok else 'FAILED'} in {time.monotonic() - started:.1f}s\n")
    return ok


def code_hash():
    """Hash of the code a run depends on, committed or not."""
    digest = hashlib.sha256()
    paths = sorted((PROJECT_ROOT / "src" / "lab").rglob("*.py"))
    paths += sorted((PROJECT_ROOT / "experiment" / "run").glob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit():
    """The checked-out commit, for reference. Not part of the run hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_config(run_dir, config):
    with (run_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def write_metrics(output_dir, metrics):
    """Save a stage's metrics in its run folder.

    Not in output/, whose files are linked into every child run.
    """
    with (output_dir.parent / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")


def load_run(run_id, stage):
    """Return the output folder of a completed run of the given stage."""
    run_dir = RUNS_DIR / run_id
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"no run {run_id} in {RUNS_DIR.relative_to(PROJECT_ROOT)}")

    config = json.loads(config_path.read_text())
    if config["status"] != "complete":
        raise SystemExit(f"{run_id} has status {config['status']}")
    if config["stage"] != stage:
        raise SystemExit(f"{run_id} is a {config['stage']} run, this needs {stage}")
    return run_dir / "output"


def link_inputs(parent_output, output_dir):
    """Link everything visible in the parent's output into a new output folder."""
    for path in parent_output.iterdir():
        target = os.path.relpath(path.resolve(), output_dir)
        (output_dir / path.name).symlink_to(target)


def execute(stage, params, parent, body, force=False):
    """Run body inside the run folder for this config. Returns the run's hash.

    parent is the hash of the run this one reads from, or None for a stage with
    no inputs. body(tee, output_dir, config) does the work and returns True on
    success; it may add result summaries to config. Exits if the run fails.
    """
    parent_output = load_run(*parent) if parent else None
    parent_id = parent[0] if parent else None

    code = code_hash()
    identity = json.dumps(
        {"stage": stage, "params": params, "parent": parent_id, "code": code},
        sort_keys=True,
    )
    run_id = hashlib.sha256(identity.encode()).hexdigest()[:12]
    run_dir = RUNS_DIR / run_id
    config_path = run_dir / "config.json"

    if config_path.exists() and not force:
        if json.loads(config_path.read_text())["status"] == "complete":
            print(f"{stage}: reusing completed run {run_id}")
            return run_id

    if run_dir.exists():
        shutil.rmtree(run_dir)
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True)
    if parent_output is not None:
        link_inputs(parent_output, output_dir)

    config = {
        "hash": run_id,
        "stage": stage,
        "parent": parent_id,
        "params": params,
        "code_hash": code,
        "git_commit": git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "running",
    }
    write_config(run_dir, config)

    started = time.monotonic()
    with (run_dir / "run.log").open("w") as log_file:
        tee = Tee(log_file)
        rule = "=" * 68
        tee.write(f"{rule}\n {stage} -- {run_id}\n parent {parent_id}, {params}\n{rule}\n")

        try:
            ok = body(tee, output_dir, config)
        except BaseException:
            config["status"] = "failed"
            write_config(run_dir, config)
            raise

        config["status"] = "complete" if ok else "failed"
        config["seconds"] = round(time.monotonic() - started, 1)
        config["outputs"] = sorted(
            path.name for path in output_dir.iterdir() if not path.is_symlink()
        )
        write_config(run_dir, config)
        tee.write(f"\n{rule}\n {config['status']}: {run_id} "
                  f"in {config['seconds']:.1f}s\n"
                  f" {run_dir.relative_to(PROJECT_ROOT)}\n{rule}\n")

    if not ok:
        raise SystemExit(1)
    return run_id


def stage_parser(description, parent_stage):
    """The command line every run_<stage>.py shares.

    A stage takes a previous stage's run with --parent, or -n to run itself and
    everything upstream from nothing.
    """
    parser = argparse.ArgumentParser(description=description)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parent", help=f"hash of a {parent_stage} run to use as input")
    source.add_argument("-n", "--objects", type=int,
                        help="generate this many objects and run every upstream stage")
    parser.add_argument("--force", action="store_true",
                        help="redo this stage if it already completed")
    return parser
