"""Project locations shared by every stage."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# experiment/run/ sets LAB_DATA_DIR so each run's stages read and write its
# own output folder. A stage run by hand uses data/.
DATA_DIR = Path(os.environ.get("LAB_DATA_DIR", PROJECT_ROOT / "data"))

RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"
