"""Calculate the allowed revolutions/path parameter k bounds from Siminski et al.

The notation and calculation follow Equation (29):

    k_min = floor(2 * Delta_t / P(a_min)) / 2
    k_max = floor(2 * Delta_t / P(a_max)) / 2

using the orbital period P(a) from Kepler's third law and the time interval
between two tracklets' attributable epochs, Delta_t = t2 - t1.

Note: period grows with semi-major axis, so P(a_min) < P(a_max). That means
the raw value computed from a_min (more, shorter-period revolutions fit in
the same Delta_t) is actually *larger* than the raw value computed from
a_max, even though the paper labels them k_min and k_max respectively. This
module computes both raw values and takes min()/max() to form the actual
[k_min, k_max] range iterated over in Eq. (30), rather than trusting the
paper's variable names literally.

k is the same parameter consumed by src/lab/model/lambert.py's
lambert_boundary_value(): its integer part is the number of full
revolutions M, and its fractional part (0 or 0.5) selects the short-way or
long-way transfer.
"""

from itertools import combinations

import numpy as np
from astropy.constants import GM_earth

from lab.paths import DATA_DIR
from lab.hypothesis.range_bounds import A_MIN, A_MAX, VARIANTS, ATTRIBUTABLES_FILES, population_for


# One k-bounds file per dataset chain, matching src/lab/hypothesis/range_bounds.py.
# The k bounds depend only on the attributable epochs t0, which the clean and
# noisy datasets share exactly, so the two files come out identical in
# practice. They are still written separately: it keeps each chain
# self-contained, and it stops being true the moment the two chains are
# generated from different observation schedules.
PRECISION_OUTPUT_FILES = {
    variant: DATA_DIR / f"k_bounds_{variant}.npz"
    for variant in VARIANTS
}

READABLE_OUTPUT_FILES = {
    variant: DATA_DIR / f"k_bounds_{variant}.txt"
    for variant in VARIANTS
}

# km^3/s^2, matches src/lab/data/new_data_gen.py's unit convention.
MU = GM_earth.value / 1_000_000_000.0


def orbital_period(a, mu=MU):
    """Orbital period from semi-major axis via Kepler's third law."""
    return 2.0 * np.pi * np.sqrt(a**3 / mu)


def calculate_k_bounds(delta_t, a_min=A_MIN, a_max=A_MAX, mu=MU):
    """Calculate k_min and k_max for a tracklet pair using Equation (29).

    Parameters
    ----------
    delta_t : float
        Time interval between the two tracklets' attributable epochs (s).
        Only the magnitude matters.
    a_min, a_max : float
        Allowed semi-major axis interval (km), Eq. (22).

    Returns
    -------
    k_min, k_max : float
        Admissible revolutions/path-parameter range, each a multiple of
        0.5, for use in the k grid of Eq. (30):
        {k_min, k_min + 0.5, ..., k_max}.
    """
    delta_t = abs(delta_t)

    period_a_min = orbital_period(a_min, mu)
    period_a_max = orbital_period(a_max, mu)

    # See module docstring: sort explicitly instead of trusting which of
    # the two raw values the paper's naming implies is smaller.
    k_at_a_min = np.floor(2.0 * delta_t / period_a_min) / 2.0
    k_at_a_max = np.floor(2.0 * delta_t / period_a_max) / 2.0

    k_min = min(k_at_a_min, k_at_a_max)
    k_max = max(k_at_a_min, k_at_a_max)

    return k_min, k_max


def k_grid(k_min, k_max):
    """Enumerate the admissible k values of Eq. (30): k_min, k_min+0.5, ..., k_max."""
    n_steps = int(round((k_max - k_min) * 2.0))
    return [k_min + 0.5 * i for i in range(n_steps + 1)]


def build_k_bounds(variant):
    """Calculate k bounds for every pair of tracklets in the attributables file.

    All unordered pairs are considered, not just pairs within the same
    object, matching the paper's all-vs-all association test (Section 6.1):
    a real association pipeline does not know in advance which tracklets
    belong together.

    Parameters
    ----------
    variant : str
        Which dataset chain to process, "clean" or "noisy".
    """
    # The same population the range bounds use. k comes from the orbital
    # period, so only the semi-major axis limits matter here -- eccentricity
    # does not enter Kepler's third law.
    a_min, a_max, _ = population_for(variant)

    with np.load(ATTRIBUTABLES_FILES[variant]) as archive:
        object_ids = archive["object_id"]
        tracklet_ids = archive["tracklet_id"]
        t0_values = archive["t0"]

    pair_rows = []

    for i, j in combinations(range(len(t0_values)), 2):
        # Stored as a magnitude, matching what calculate_k_bounds actually
        # uses -- combinations() does not guarantee t0[i] < t0[j], and a
        # signed value here would misleadingly suggest k_min/k_max (always
        # >= 0) were computed from a negative time interval.
        delta_t = abs(t0_values[j] - t0_values[i])

        # Two tracklets at the same epoch cannot be associated via
        # Lambert's problem, which requires two distinct epochs. This can
        # legitimately happen once multiple objects are simulated: every
        # object shares the same start_epoch, so e.g. object 0's tracklet 0
        # and object 1's tracklet 0 land at identical absolute times.
        if delta_t == 0.0:
            continue

        k_min, k_max = calculate_k_bounds(delta_t, a_min, a_max)

        pair_rows.append([
            object_ids[i], tracklet_ids[i],
            object_ids[j], tracklet_ids[j],
            delta_t, k_min, k_max,
        ])

    pair_rows = np.asarray(pair_rows, dtype=np.float64)

    np.savez(
        PRECISION_OUTPUT_FILES[variant],
        object_id_1=pair_rows[:, 0].astype(int),
        tracklet_id_1=pair_rows[:, 1].astype(int),
        object_id_2=pair_rows[:, 2].astype(int),
        tracklet_id_2=pair_rows[:, 3].astype(int),
        delta_t_s=pair_rows[:, 4],
        k_min=pair_rows[:, 5],
        k_max=pair_rows[:, 6],
        a_min_km=a_min,
        a_max_km=a_max,
    )

    np.savetxt(
        READABLE_OUTPUT_FILES[variant],
        pair_rows,
        delimiter=",",
        fmt=["%d", "%d", "%d", "%d", "%.3f", "%.1f", "%.1f"],
        header=(
            "object_id_1,tracklet_id_1,object_id_2,tracklet_id_2,"
            "delta_t_s,k_min,k_max"
        ),
        comments="",
    )

    for row in pair_rows:
        obj1, trk1, obj2, trk2, delta_t, k_min, k_max = row
        grid = k_grid(k_min, k_max)
        print(
            f"Object {int(obj1)} tracklet {int(trk1)} <-> "
            f"Object {int(obj2)} tracklet {int(trk2)}: "
            f"delta_t={delta_t:.1f}s, k in {grid}"
        )

    print(f"Wrote precise bounds to {PRECISION_OUTPUT_FILES[variant]}")
    print(f"Wrote readable bounds to {READABLE_OUTPUT_FILES[variant]}")


def main():
    """Calculate k bounds for the clean and the noisy chain."""
    for variant in VARIANTS:
        if not ATTRIBUTABLES_FILES[variant].exists():
            print(f"Skipping {variant}: {ATTRIBUTABLES_FILES[variant]} does not exist")
            continue

        print(f"--- {variant} ---")
        build_k_bounds(variant)
        print()


if __name__ == "__main__":
    main()
