
import numpy as np

from lab.data.dataset import VARIANTS, data_file, load_data, uncertainties_arcsec
from lab.paths import DATA_DIR
from lab.observation.fit_attributes import fit_attributes


# All chains come from one set of orbits, stations and observation epochs. The
# only columns that differ between them are the measured RA/Dec pair, so each
# is fitted here by the same code with the same tracklet grouping, and their
# attributables stay directly comparable row for row.
#
# Each entry is (label, attributables npz, attributables txt).
DATASETS = tuple(
    (
        variant,
        DATA_DIR / f"generated_attributables_{variant}.npz",
        DATA_DIR / f"generated_attributables_{variant}.txt",
    )
    for variant in VARIANTS
)


def build_attributables(
    variant,
    attributables_npz_file,
    attributables_txt_file,
):
    """Fit one attributable per tracklet in a generated dataset.

    Parameters
    ----------
    variant : str
        The dataset chain to fit, one of VARIANTS.

    attributables_npz_file : Path
        Destination for the machine-readable attributables.

    attributables_txt_file : Path
        Destination for the human-readable attributables.
    """
    # Load the simulated observations and their angular-error settings.
    #
    # The input data matrix is created by src/lab/data/gcrs_data_generation.py. The
    # columns used here are:
    #   6  = measured right ascension, alpha (rad)
    #   7  = measured declination, delta (rad)
    #   14 = observation time (s)
    #   15 = tracklet ID
    #   16 = object ID
    data = load_data(variant)

    # Relative error represents point-to-point scatter inside a tracklet.
    # Systematic error represents a common angular offset for the tracklet.
    # Both values are stored in arcseconds and converted to radians by
    # fit_attributes().
    #
    # The clean dataset carries zeros here, because no noise was added to it.
    # Its fitted attributables are still exact, but every covariance it
    # produces is the zero matrix, so the clean file cannot be used by stages
    # that invert Ca.
    relative_error, systematic_error = uncertainties_arcsec(variant)

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
        attributables_npz_file,
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
        attributables_txt_file,
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

    return len(results)


def main():
    """Fit attributables for every dataset chain."""
    for label, npz_file, txt_file in DATASETS:
        # Each dataset is optional: deleting one to regenerate it should not
        # stop the others from being fitted.
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
