from pathlib import Path

import numpy as np

from fit_attributes import fit_attributes

THIS_DIR = Path(__file__).resolve().parent
GENERATED_DATA_FILE = THIS_DIR.parent / "simulation" / "generated_data.npz"
ATTRIBUTABLES_NPZ_FILE = THIS_DIR / "generated_attributables.npz"
ATTRIBUTABLES_TXT_FILE = THIS_DIR / "generated_attributables.txt"


def main():
    # Load the simulated observations and their angular-error settings.
    #
    # The input data matrix is created by simulation/new_data_gen.py. The
    # columns used here are:
    #   6  = measured right ascension, alpha (rad)
    #   7  = measured declination, delta (rad)
    #   14 = observation time (s)
    #   15 = tracklet ID
    #   16 = object ID
    generated = np.load(GENERATED_DATA_FILE)

    data = generated["data"]

    # Relative error represents point-to-point scatter inside a tracklet.
    # Systematic error represents a common angular offset for the tracklet.
    # Both values are stored in arcseconds and converted to radians by
    # fit_attributes().
    relative_error = generated["relative_uncertainty_arcsec"]
    systematic_error = generated["systematic_uncertainty_arcsec"]

    # IDs are stored in the numeric data matrix, so cast them back to integers.
    object_ids = data[:, 16].astype(int)
    tracklet_ids = data[:, 15].astype(int)

    # Each result will contain:
    #   [object_id, tracklet_id, t0, alpha, alpha_dot, delta, delta_dot]
    #
    # Each matching covariance is a 4x4 matrix in attributable order:
    #   [alpha, alpha_dot, delta, delta_dot]
    results = []
    covariances = []

    # Process each unique object/tracklet pair once. Sorting makes the output
    # deterministic even if rows in the input file are not grouped by tracklet.
    for object_id, tracklet_id in sorted(set(zip(object_ids, tracklet_ids))):
        # Select every angular observation belonging to this tracklet.
        tracklet = data[
            (object_ids == object_id)
            & (tracklet_ids == tracklet_id)
        ]

        # Compress the series of RA/Dec observations into one attributable:
        #   a  = [alpha, alpha_dot, delta, delta_dot]
        #   Ca = covariance of a
        #   t0 = mean observation time used as the fit reference epoch
        a, Ca, t0 = fit_attributes(
            tracklet[:, 14],
            tracklet[:, 6],
            tracklet[:, 7],
            relative_error,
            systematic_error,
        )

        results.append([object_id, tracklet_id, t0, *a])
        covariances.append(Ca)

    # Convert the accumulated Python lists into arrays for slicing and saving.
    # Shapes are (number_of_tracklets, 7) and (number_of_tracklets, 4, 4).
    results = np.array(results)
    covariances = np.array(covariances)

    # Save the machine-readable form without flattening the covariance matrices.
    # This is the preferred file for later numerical processing because it
    # preserves integer IDs and the original array shapes.
    np.savez(
        ATTRIBUTABLES_NPZ_FILE,
        object_id=results[:, 0].astype(int),
        tracklet_id=results[:, 1].astype(int),
        t0=results[:, 2],
        attributable=results[:, 3:7],
        Ca=covariances,
    )

    # Build a two-dimensional table for the human-readable text file. Each 4x4
    # covariance matrix is flattened in row-major order:
    #   Ca_00, Ca_01, ..., Ca_33.
    #
    # The final table has 23 columns:
    #   2 IDs + t0 + 4 attributable values + 16 covariance values.
    readable_results = np.column_stack([
        results,
        covariances.reshape(len(covariances), 16),
    ])

    # Write a CSV-style text representation. The header names specify the units
    # of t0 and the attributable values; covariance units follow from the pair
    # of attributable components represented by each matrix entry.
    np.savetxt(
        ATTRIBUTABLES_TXT_FILE,
        readable_results,
        delimiter=",",
        fmt=[
            "%d",           # object_id: readable integer
            "%d",           # tracklet_id: readable integer
            "%.3f",         # t0, seconds
            "%.12f",        # alpha, radians
            "%.12e",        # alpha_dot, radians/second
            "%.12f",        # delta, radians
            "%.12e",        # delta_dot, radians/second
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


if __name__ == "__main__":
    main()
