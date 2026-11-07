"""Reading the dataset gcrs_data_generation.py writes.

One text file of rows per chain. Chains share orbits, stations and epochs and
differ only in RA/Dec. A JSON record holds the settings rows cannot. The text
files are canonical and read back exactly. Paths resolve DATA_DIR when called.
"""

import json

import numpy as np

import lab.paths

# Siminski et al. (2014) Section 6.1; one noisy chain per level.
RELATIVE_NOISE_CASES_ARCSEC = [0.7, 1.2, 2.0]  # arcsec

NOISY_VARIANTS = {
    f"noisy_{sigma_arcsec}as": sigma_arcsec
    for sigma_arcsec in RELATIVE_NOISE_CASES_ARCSEC
}

# Every stage runs each of these as an independent chain.
VARIANTS = ("clean", *NOISY_VARIANTS)

# Row layout. Index rows by these names; test/test_dataset.py checks them
# against the generator.
SATELLITE_POSITION = slice(0, 3)    # km, truth: metrics only
SATELLITE_VELOCITY = slice(3, 6)    # km/s, truth: metrics only
RA_COLUMN = 6                       # rad, measured
DEC_COLUMN = 7                      # rad, measured
SITE_POSITION = slice(8, 11)        # km, GCRS
SITE_VELOCITY = slice(11, 14)       # km/s, GCRS
TIME_COLUMN = 14                    # s since the epoch
TRACKLET_ID_COLUMN = 15
OBJECT_ID_COLUMN = 16
OBSERVATION_ID_COLUMN = 17
NUM_COLUMNS = 18


def data_file(variant):
    return lab.paths.DATA_DIR / f"generated_data_gcrs_{variant}.txt"


def settings_file():
    return lab.paths.DATA_DIR / "generated_data_gcrs_settings.json"


def load_data(variant):
    """The observation rows of one chain, in the layout above.

    Raises if the file has a different column count, which means it came from
    an older generator.
    """
    data = np.loadtxt(data_file(variant), delimiter=",", ndmin=2)
    if data.shape[1] != NUM_COLUMNS:
        raise ValueError(
            f"{data_file(variant)} has {data.shape[1]} columns, expected "
            f"{NUM_COLUMNS}; it was written by an older generator, regenerate it."
        )
    return data


def tracklet_ids(data):
    """Every (object_id, tracklet_id) in the rows, sorted."""
    return sorted(set(zip(
        data[:, OBJECT_ID_COLUMN].astype(int).tolist(),
        data[:, TRACKLET_ID_COLUMN].astype(int).tolist(),
    )))


def tracklet_rows(data, object_id, tracklet_id):
    """One tracklet's rows, in time order.

    Sorted because interpolating the site to t0 (np.interp) needs increasing
    times, and the file order is the generator's choice, not a guarantee.
    """
    rows = data[
        (data[:, OBJECT_ID_COLUMN].astype(int) == object_id)
        & (data[:, TRACKLET_ID_COLUMN].astype(int) == tracklet_id)
    ]
    if len(rows) == 0:
        raise KeyError(f"no rows for object {object_id} tracklet {tracklet_id}")
    return rows[np.argsort(rows[:, TIME_COLUMN], kind="stable")]


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
