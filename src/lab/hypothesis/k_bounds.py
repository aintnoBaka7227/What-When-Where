"""Stage 4: revolution-count bounds per tracklet pair, Siminski et al. Eq. (29).

    k = floor(2 dt / P(a)) / 2,   at a = a_min and a = a_max

The paper's labels are reversed: P(a_min) is shorter, so a_min gives the
larger k. Both values are computed and sorted.

k's integer part is the number of full revolutions; its fraction (0 or 0.5)
selects the short-way or long-way transfer (see lambert.py).
"""

from itertools import combinations

import numpy as np
from astropy.constants import GM_earth

from lab.paths import VariantFiles
from lab.data.dataset import VARIANTS
from lab.hypothesis.range_bounds import A_MIN, A_MAX, ATTRIBUTABLES_FILES, population_for


# k depends only on t0, which every chain shares, so these files are identical.
# Each chain still gets its own, to stay self-contained.
PRECISION_OUTPUT_FILES = VariantFiles("k_bounds_{variant}.npz", VARIANTS)
READABLE_OUTPUT_FILES = VariantFiles("k_bounds_{variant}.txt", VARIANTS)

# km^3/s^2, as in the generator.
MU = GM_earth.value / 1_000_000_000.0


def orbital_period(a, mu=MU):
    """Kepler's third law."""
    return 2.0 * np.pi * np.sqrt(a**3 / mu)


def calculate_k_bounds(delta_t, a_min=A_MIN, a_max=A_MAX, mu=MU):
    """Eq. (29): (k_min, k_max), multiples of 0.5, for |delta_t| in s."""
    delta_t = abs(delta_t)

    period_a_min = orbital_period(a_min, mu)
    period_a_max = orbital_period(a_max, mu)

    # Sorted below: the paper's naming has them reversed.
    k_at_a_min = np.floor(2.0 * delta_t / period_a_min) / 2.0
    k_at_a_max = np.floor(2.0 * delta_t / period_a_max) / 2.0

    k_min = min(k_at_a_min, k_at_a_max)
    k_max = max(k_at_a_min, k_at_a_max)

    return k_min, k_max


def k_grid(k_min, k_max):
    """Eq. (30)'s k grid: k_min, k_min + 0.5, ..., k_max."""
    n_steps = int(round((k_max - k_min) * 2.0))
    return [k_min + 0.5 * i for i in range(n_steps + 1)]


def build_k_bounds(variant):
    """k bounds for every unordered tracklet pair of one chain.

    All pairs, not just same-object ones: the paper's all-vs-all test
    (Section 6.1) is not told which tracklets belong together.
    """
    # Only a matters: eccentricity does not enter the period.
    a_min, a_max, _ = population_for(variant)

    with np.load(ATTRIBUTABLES_FILES[variant]) as archive:
        object_ids = archive["object_id"]
        tracklet_ids = archive["tracklet_id"]
        t0_values = archive["t0"]

    pair_rows = []

    for i, j in combinations(range(len(t0_values)), 2):
        # A magnitude: combinations() does not order the epochs.
        delta_t = abs(t0_values[j] - t0_values[i])

        # Lambert needs two distinct epochs. Per-object start offsets make
        # this unreachable for generated data; kept as a guard.
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
    """k bounds for every chain whose attributables exist."""
    for variant in VARIANTS:
        if not ATTRIBUTABLES_FILES[variant].exists():
            print(f"Skipping {variant}: {ATTRIBUTABLES_FILES[variant]} does not exist")
            continue

        print(f"--- {variant} ---")
        build_k_bounds(variant)
        print()


if __name__ == "__main__":
    main()
