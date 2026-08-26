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

# simulation/new_data_gen.py generates two datasets from one orbit, one set of
# stations and one set of epochs: "clean" carries the exact angles, "noisy"
# carries the same angles with Siminski measurement error applied. Every stage
# from here down runs both as two independent chains, so a result can always be
# compared against its own noise-free counterpart.
#
# This module is the path authority for the whole pipeline: the later stages
# import these dictionaries rather than building paths themselves.
VARIANTS = ("clean", "noisy")

GENERATED_DATA_FILES = {
    variant: THIS_DIR.parent / "simulation" / f"generated_data_{variant}.npz"
    for variant in VARIANTS
}

ATTRIBUTABLES_FILES = {
    variant: THIS_DIR.parent / "observation" / f"generated_attributables_{variant}.npz"
    for variant in VARIANTS
}

PRECISION_OUTPUT_FILES = {
    variant: THIS_DIR / f"range_bounds_{variant}.npz"
    for variant in VARIANTS
}

READABLE_OUTPUT_FILES = {
    variant: THIS_DIR / f"range_bounds_{variant}.txt"
    for variant in VARIANTS
}

# Fallback population limits, for datasets generated before
# simulation/new_data_gen.py began recording its own. Everything below
# goes through population_for() instead: E_MAX = 0.01 describes test case
# 'a' alone, and applying it to an eccentric population silently puts the
# true range outside the region the optimizer may search.
A_MIN = 38_000.0
A_MAX = 45_000.0
E_MAX = 0.01


def population_for(variant):
    """The orbit population a dataset was actually generated from.

    simulation/new_data_gen.py records the semi-major axis and eccentricity
    limits its test case sampled, so the admissible region (Eqs. 23-24)
    follows the data rather than being pinned to one case. Datasets written
    before those keys existed fall back to the module constants.

    Returns
    -------
    a_min, a_max, e_max : float
    """
    with np.load(GENERATED_DATA_FILES[variant]) as archive:
        if "a_min_km" not in archive:
            return A_MIN, A_MAX, E_MAX

        return (
            float(archive["a_min_km"]),
            float(archive["a_max_km"]),
            float(archive["e_max"]),
        )


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


def v_s_i_at_epoch(
    data,
    object_id,
    tracklet_id,
    t0_i,
):
    """Obtain the observing-site velocity v_s_i at attributable epoch t0_i.

    Sibling to r_s_i_at_epoch, used by modelled/model_rates.py to form
    the topocentric relative velocity rho_dot = v_sat - v_s needed for
    Eq. (17)'s modeled angular rates. Not needed by this file's own range
    bounds (Eqs. 26-28), which only use site position.
    """
    object_ids = data[:, 16].astype(int)
    tracklet_ids = data[:, 15].astype(int)

    tracklet_i = data[
        (object_ids == object_id)
        & (tracklet_ids == tracklet_id)
    ]

    order = np.argsort(tracklet_i[:, 14])
    times_i = tracklet_i[order, 14]
    v_s_observations_i = tracklet_i[order, 11:14]

    v_s_i = np.array([
        np.interp(
            t0_i,
            times_i,
            v_s_observations_i[:, axis],
        )
        for axis in range(3)
    ])

    return v_s_i


def build_range_bounds(variant):
    """Calculate the Siminski range bounds for every fitted tracklet.

    Parameters
    ----------
    variant : str
        Which dataset chain to process, "clean" or "noisy". Only the
        attributables differ between the two -- the site positions are
        shared -- but each chain writes its own bounds file so the stages
        below never have to know which one they are working on.
    """
    with np.load(GENERATED_DATA_FILES[variant]) as archive:
        data = archive["data"]

    with np.load(ATTRIBUTABLES_FILES[variant]) as archive:
        object_ids = archive["object_id"]
        tracklet_ids = archive["tracklet_id"]
        t0_values = archive["t0"]
        attributables = archive["attributable"]

    a_min, a_max, e_max = population_for(variant)
    r_min, r_max = calculate_r_min_r_max(a_min, a_max, e_max)

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
        PRECISION_OUTPUT_FILES[variant],
        object_id=object_ids,
        tracklet_id=tracklet_ids,
        t0=t0_values,
        r_s_i_km=r_s_i_array,
        u_i=u_i_array,
        c_i_km=c_i_array,
        rho_min_i_km=rho_min_i_array,
        rho_max_i_km=rho_max_i_array,
        a_min_km=a_min,
        a_max_km=a_max,
        e_max=e_max,
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
        READABLE_OUTPUT_FILES[variant],
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
        f"population a in [{a_min:.0f}, {a_max:.0f}] km, e <= {e_max}"
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

    print(f"Wrote precise bounds to {PRECISION_OUTPUT_FILES[variant]}")
    print(f"Wrote readable bounds to {READABLE_OUTPUT_FILES[variant]}")


def available_variants():
    """The variants whose inputs are both present on disk.

    Each stage skips a variant it cannot run rather than failing outright,
    so regenerating one chain never blocks the other.
    """
    return [
        variant
        for variant in VARIANTS
        if GENERATED_DATA_FILES[variant].exists()
        and ATTRIBUTABLES_FILES[variant].exists()
    ]


def main():
    """Calculate range bounds for the clean and the noisy chain."""
    for variant in VARIANTS:
        if variant not in available_variants():
            print(f"Skipping {variant}: inputs missing")
            continue

        print(f"--- {variant} ---")
        build_range_bounds(variant)
        print()


if __name__ == "__main__":
    main()
