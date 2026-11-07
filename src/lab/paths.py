"""Project locations shared by every stage."""

import os
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# experiment/run/ sets LAB_DATA_DIR so each run's stages read and write its
# own output folder. A stage run by hand uses data/.
DATA_DIR = Path(os.environ.get("LAB_DATA_DIR", PROJECT_ROOT / "data"))

RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"


class VariantFiles(Mapping):
    """{variant: DATA_DIR / pattern}, reading DATA_DIR at each lookup.

    So repointing DATA_DIR after import moves every stage with it. A dict
    built at import would keep pointing at data/.
    """

    def __init__(self, pattern, variants):
        self._pattern = pattern
        self._variants = tuple(variants)

    def __getitem__(self, variant):
        if variant not in self._variants:
            raise KeyError(variant)
        # A module global, so this reads DATA_DIR as it is now.
        return DATA_DIR / self._pattern.format(variant=variant)

    def __iter__(self):
        return iter(self._variants)

    def __len__(self):
        return len(self._variants)

    def __repr__(self):
        return f"VariantFiles({self._pattern!r} in {DATA_DIR})"
