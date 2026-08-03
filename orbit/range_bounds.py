"""Calculate the topocentric-range bounds from Siminski et al.

The notation and calculations follow Equations (23), (24), and (26)-(28):

    ||r_min|| = a_min * (1 - e_max)
    ||r_max|| = a_max * (1 + e_max)
    c_i       = r_s_i dot u_i

    rho_min_i = -c_i + sqrt(
        c_i**2 + ||r_min||**2 - ||r_s_i||**2
    )

    rho_max_i = -c_i + sqrt(
        c_i**2 + ||r_max||**2 - ||r_s_i||**2
    )

Here, rho_i is the topocentric range from the observing site to the object.
"""

from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERATED_DATA_FILE = THIS_DIR.parent / "simulation" / "generated_data.npz"
ATTRIBUTABLES_FILE = THIS_DIR.parent / "observation" / "generated_attributables.npz"
PRECISION_OUTPUT_FILE = THIS_DIR / "range_bounds.npz"
READABLE_OUTPUT_FILE = THIS_DIR / "range_bounds.txt"

# Near-GEO population limits used in the paper's test case.
A_MIN = 38_000.0
A_MAX = 45_000.0
E_MAX = 0.01


def calculate_r_min_r_max(
    a_min=A_MIN,
    a_max=A_MAX,
    e_max=E_MAX,
):
    """Calculate ||r_min|| and ||r_max|| using Equations (23) and (24)."""
    r_min = a_min * (1.0 - e_max)
    r_max = a_max * (1.0 + e_max)

    return r_min, r_max


def calculate_u_i(alpha_i, delta_i):
    """Calculate the line-of-sight unit vector u_i from alpha_i and delta_i."""
    u_i = np.array([
        np.cos(delta_i) * np.cos(alpha_i),
        np.cos(delta_i) * np.sin(alpha_i),
        np.sin(delta_i),
    ])

    return u_i


def calculate_rho_bounds_i(
    r_s_i,
    u_i,
    r_min,
    r_max,
):
    """Calculate rho_min_i and rho_max_i using Equations (26)-(28)."""
    c_i = np.dot(r_s_i, u_i)

    rho_min_i = (
        -c_i
        + np.sqrt(
            c_i**2
            + r_min**2
            - np.dot(r_s_i, r_s_i)
        )
    )

    rho_max_i = (
        -c_i
        + np.sqrt(
            c_i**2
            + r_max**2
            - np.dot(r_s_i, r_s_i)
        )
    )

    return c_i, rho_min_i, rho_max_i


def r_s_i_at_epoch(
    data,
    object_id,
    tracklet_id,
    t0_i,
):
    """Obtain the observing-site position r_s_i at attributable epoch t0_i."""
    object_ids = data[:, 16].astype(int)
    tracklet_ids = data[:, 15].astype(int)

    tracklet_i = data[
        (object_ids == object_id)
        & (tracklet_ids == tracklet_id)
    ]

    order = np.argsort(tracklet_i[:, 14])
    times_i = tracklet_i[order, 14]
    r_s_observations_i = tracklet_i[order, 8:11]

    r_s_i = np.array([
        np.interp(
            t0_i,
            times_i,
            r_s_observations_i[:, axis],
        )
        for axis in range(3)
    ])

    return r_s_i


def main():
    """Calculate the Siminski range bounds for every fitted tracklet."""
    with np.load(GENERATED_DATA_FILE) as archive:
        data = archive["data"]

    with np.load(ATTRIBUTABLES_FILE) as archive:
        object_ids = archive["object_id"]
        tracklet_ids = archive["tracklet_id"]
        t0_values = archive["t0"]
        attributables = archive["attributable"]

    r_min, r_max = calculate_r_min_r_max()

    saved_r_s_i = []
    saved_u_i = []
    saved_c_i = []
    saved_rho_min_i = []
    saved_rho_max_i = []

    for object_id, tracklet_id, t0_i, attributable_i in zip(
        object_ids,
        tracklet_ids,
        t0_values,
        attributables,
    ):
        # Paper attributable ordering:
        # A_i = [alpha_i, alpha_dot_i, delta_i, delta_dot_i]
        alpha_i = attributable_i[0]
        delta_i = attributable_i[2]

        u_i = calculate_u_i(alpha_i, delta_i)

        r_s_i = r_s_i_at_epoch(
            data,
            object_id,
            tracklet_id,
            t0_i,
        )

        c_i, rho_min_i, rho_max_i = calculate_rho_bounds_i(
            r_s_i,
            u_i,
            r_min,
            r_max,
        )

        saved_r_s_i.append(r_s_i)
        saved_u_i.append(u_i)
        saved_c_i.append(c_i)
        saved_rho_min_i.append(rho_min_i)
        saved_rho_max_i.append(rho_max_i)

    r_s_i_array = np.asarray(saved_r_s_i)
    u_i_array = np.asarray(saved_u_i)
    c_i_array = np.asarray(saved_c_i)
    rho_min_i_array = np.asarray(saved_rho_min_i)
    rho_max_i_array = np.asarray(saved_rho_max_i)

    # Full-precision output for later optimisation.
    np.savez(
        PRECISION_OUTPUT_FILE,
        object_id=object_ids,
        tracklet_id=tracklet_ids,
        t0=t0_values,
        r_s_i_km=r_s_i_array,
        u_i=u_i_array,
        c_i_km=c_i_array,
        rho_min_i_km=rho_min_i_array,
        rho_max_i_km=rho_max_i_array,
        a_min_km=A_MIN,
        a_max_km=A_MAX,
        e_max=E_MAX,
        r_min_km=r_min,
        r_max_km=r_max,
    )

    # Readable output for inspection.
    readable_rows = np.column_stack((
        object_ids,
        tracklet_ids,
        t0_values,
        r_s_i_array,
        u_i_array,
        c_i_array,
        rho_min_i_array,
        rho_max_i_array,
    ))

    np.savetxt(
        READABLE_OUTPUT_FILE,
        readable_rows,
        delimiter=",",
        fmt=[
            "%d",
            "%d",
            "%.3f",
            "%.6f",
            "%.6f",
            "%.6f",
            "%.12f",
            "%.12f",
            "%.12f",
            "%.6f",
            "%.6f",
            "%.6f",
        ],
        header=(
            "object_id,tracklet_id,t0_s,"
            "r_s_i_x_km,r_s_i_y_km,r_s_i_z_km,"
            "u_i_x,u_i_y,u_i_z,"
            "c_i_km,rho_min_i_km,rho_max_i_km"
        ),
        comments="",
    )

    print(
        f"||r_min|| = {r_min:.3f} km, "
        f"||r_max|| = {r_max:.3f} km"
    )

    for object_id, tracklet_id, rho_min_i, rho_max_i in zip(
        object_ids,
        tracklet_ids,
        rho_min_i_array,
        rho_max_i_array,
    ):
        print(
            f"Object {object_id}, tracklet {tracklet_id}: "
            f"rho_min_i = {rho_min_i:.3f} km, "
            f"rho_max_i = {rho_max_i:.3f} km"
        )

    print(f"Wrote precise bounds to {PRECISION_OUTPUT_FILE}")
    print(f"Wrote readable bounds to {READABLE_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
