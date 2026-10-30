"""Print the metrics of a run and of every run upstream of it.

    uv run python experiment/run/show_metrics.py <hash>
"""

import argparse
import json

from lab.paths import RUNS_DIR

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("hash", help="hash of a run")
run_id = parser.parse_args().hash

while run_id is not None:
    run_dir = RUNS_DIR / run_id
    config = json.loads((run_dir / "config.json").read_text())
    print(f"--- {config['stage']} {run_id}")
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(metrics_path.read_text().rstrip())
    run_id = config["parent"]
