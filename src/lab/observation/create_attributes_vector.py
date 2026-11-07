"""Stage 2: fit one attributable per tracklet, for every chain."""

import numpy as np

from lab.data.dataset import (
    DEC_COLUMN,
    RA_COLUMN,
    TIME_COLUMN,
    VARIANTS,
    data_file,
    load_data,
    tracklet_ids,
    tracklet_rows,
    uncertainties_arcsec,
)
from lab.paths import VariantFiles
from lab.observation.fit_attributes import fit_attributes


# Stage 2's own copy of these paths, so it does not import stage 3.
ATTRIBUTABLES_NPZ_FILES = VariantFiles("generated_attributables_{variant}.npz", VARIANTS)
ATTRIBUTABLES_TXT_FILES = VariantFiles("generated_attributables_{variant}.txt", VARIANTS)


def build_attributables(
    variant,
    attributables_npz_file,
    attributables_txt_file,
):
    """Fit every tracklet of one chain and write the .npz (canonical) and .txt."""
    data = load_data(variant)

    # Zero for clean, which then gets a zero covariance: no stage may invert it.
    relative_error, systematic_error = uncertainties_arcsec(variant)

    # Rows of [object_id, tracklet_id, t0, alpha, alpha_dot, delta, delta_dot].
    results = []
    covariances = []

    for object_id, tracklet_id in tracklet_ids(data):
        # Time-sorted, so the output does not depend on file row order.
        tracklet = tracklet_rows(data, object_id, tracklet_id)

        a, Ca, t0 = fit_attributes(
            tracklet[:, TIME_COLUMN],
            tracklet[:, RA_COLUMN],
            tracklet[:, DEC_COLUMN],
            relative_error,
            systematic_error,
        )

        results.append([object_id, tracklet_id, t0, *a])
        covariances.append(Ca)

    results = np.array(results)
    covariances = np.array(covariances)

    np.savez(
        attributables_npz_file,
        object_id=results[:, 0].astype(int),
        tracklet_id=results[:, 1].astype(int),
        t0=results[:, 2],
        attributable=results[:, 3:7],
        Ca=covariances,
    )

    # Readable copy: Ca flattened row-major, 23 columns in all.
    readable_results = np.column_stack([
        results,
        covariances.reshape(len(covariances), 16),
    ])

    np.savetxt(
        attributables_txt_file,
        readable_results,
        delimiter=",",
        fmt=[
            "%d",           # object_id
            "%d",           # tracklet_id
            "%.3f",         # t0, s
            "%.12f",        # alpha, rad
            "%.12e",        # alpha_dot, rad/s
            "%.12f",        # delta, rad
            "%.12e",        # delta_dot, rad/s
            "%.6e", "%.6e", "%.6e", "%.6e",
            "%.6e", "%.6e", "%.6e", "%.6e",
            "%.6e", "%.6e", "%.6e", "%.6e",
            "%.6e", "%.6e", "%.6e", "%.6e",
        ],
        header=(
            "object_id,tracklet_id,t0_s,"
            "alpha_rad,alpha_dot_rad_s,"
            "delta_rad,delta_dot_rad_s,"
            "Ca_00,Ca_01,Ca_02,Ca_03,"
            "Ca_10,Ca_11,Ca_12,Ca_13,"
            "Ca_20,Ca_21,Ca_22,Ca_23,"
            "Ca_30,Ca_31,Ca_32,Ca_33"
        ),
        comments="",
    )

    return len(results)


def main():
    """Fit every chain whose data file exists."""
    for label in VARIANTS:
        npz_file = ATTRIBUTABLES_NPZ_FILES[label]
        txt_file = ATTRIBUTABLES_TXT_FILES[label]
        if not data_file(label).exists():
            print(f"Skipping {label}: {data_file(label)} does not exist")
            continue

        tracklet_count = build_attributables(
            label,
            npz_file,
            txt_file,
        )

        print(f"Fitted {tracklet_count} {label} tracklets")
        print(f"  Wrote precise attributables to {npz_file}")
        print(f"  Wrote readable attributables to {txt_file}")


if __name__ == "__main__":
    main()
