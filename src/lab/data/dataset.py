"""The dataset gcrs_data_generation.py writes, as the later stages read it.

Every chain is one text file of observation rows: clean holds the true angles,
and each noisy file adds measurement error at one relative noise level. They
come from the same orbits, stations and epochs and differ only in the RA/Dec
columns, so a result can be compared across noise levels. Settings the rows
cannot hold, such as the noise and orbit population, are in one JSON record.

The text files are the canonical data. Angles are written at full precision and
read back exactly.

Paths are looked up when called, not at import, so that a run script can point
lab.paths.DATA_DIR at its own output folder.
"""

import json

import numpy as np

import lab.paths

# Relative angular noise levels tested by Siminski et al. (2014), Section 6.1:
# sigma_alpha,delta in {0.7", 1.2", 2"}. Every run writes one noisy file per level.
RELATIVE_NOISE_CASES_ARCSEC = [0.7, 1.2, 2.0]  # arcsec

NOISY_VARIANTS = {
    f"noisy_{sigma_arcsec}as": sigma_arcsec
    for sigma_arcsec in RELATIVE_NOISE_CASES_ARCSEC
}

# Every stage runs each of these as an independent chain.
VARIANTS = ("clean", *NOISY_VARIANTS)


def data_file(variant):
    return lab.paths.DATA_DIR / f"generated_data_gcrs_{variant}.txt"


def settings_file():
    return lab.paths.DATA_DIR / "generated_data_gcrs_settings.json"


def load_data(variant):
    """The observation rows of one chain; see NUM_OF_OUTPUT_COLUMNS for the layout."""
    return np.loadtxt(data_file(variant), delimiter=",", ndmin=2)


def load_settings():
    """The run-independent generation settings."""
    with settings_file().open() as handle:
        return json.load(handle)["settings"]


def uncertainties_arcsec(variant):
    """(relative, systematic) measurement uncertainty of one chain.

    Both are zero for clean, because no noise was added to it.
    """
    if variant == "clean":
        return 0.0, 0.0
    return NOISY_VARIANTS[variant], load_settings()["systematic_noise_arcsec"]


def population():
    """The orbit population the dataset was generated from: (a_min, a_max, e_max)."""
    settings = load_settings()
    a_min, a_max = settings["semi_major_axis_range_km"]
    return float(a_min), float(a_max), float(settings["eccentricity_range"][1])
