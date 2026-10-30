"""Print every run, oldest first.

    uv run python experiment/run/list_runs.py
"""

import json

from lab.paths import RUNS_DIR

configs = [json.loads(path.read_text()) for path in RUNS_DIR.glob("*/config.json")]
for config in sorted(configs, key=lambda c: c["started_utc"]):
    print(f"{config['hash']}  {config['started_utc']}  {config['status']:9s}"
          f"{config['stage']:12s} parent={config['parent']} {config['params']}")
