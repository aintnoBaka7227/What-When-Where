"""Stage 3: topocentric range bounds, Siminski et al. Eqs. (23)-(28).

    ||r_min|| = a_min (1 - e_max)          ||r_max|| = a_max (1 + e_max)
    c_i = r_s_i . u_i
    rho_i = -c_i + sqrt(c_i^2 + ||r||^2 - ||r_s_i||^2)

Also the path registry for the derived files, which later stages import.
"""


import numpy as np

from lab.data.dataset import (
    SITE_POSITION,
    SITE_VELOCITY,
    TIME_COLUMN,
    VARIANTS,
    data_file,
    load_data,
    population,
    tracklet_rows,
)
from lab.paths import VariantFiles

# Resolved in lab.paths.DATA_DIR when indexed, not at import.
ATTRIBUTABLES_FILES = VariantFiles("generated_attributables_{variant}.npz", VARIANTS)
PRECISION_OUTPUT_FILES = VariantFiles("range_bounds_{variant}.npz", VARIANTS)
ASSOCIATION_FILES = VariantFiles("association_{variant}.npz", VARIANTS)
READABLE_OUTPUT_FILES = VariantFiles("range_bounds_{variant}.txt", VARIANTS)

# Defaults only. Stages use population_for(): E_MAX = 0.01 fits test case 'a'
# alone, and would put an eccentric object's true range outside the bounds.
A_MIN = 38_000.0
A_MAX = 45_000.0
E_MAX = 0.01


def population_for(variant):
    """(a_min, a_max, e_max) the dataset was generated from, shared by all chains."""
    return population()


def calculate_r_min_r_max(
    a_min=A_MIN,
    a_max=A_MAX,
    e_max=E_MAX,
):
    """Eqs. (23)-(24)."""
    r_min = a_min * (1.0 - e_max)
    r_max = a_max * (1.0 + e_max)

    return r_min, r_max


def calculate_u_i(alpha_i, delta_i):
    """Line-of-sight unit vector from RA and Dec."""
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
    """Eqs. (26)-(28): (c_i, rho_min_i, rho_max_i)."""
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
    """Site position at t0_i, interpolated linearly.

    Exact for the generator's evenly sampled tracklets, whose mean time is the
    middle observation.
    """
    return _site_at_epoch(data, object_id, tracklet_id, t0_i, SITE_POSITION)


def _site_at_epoch(data, object_id, tracklet_id, t0_i, columns):
    """Interpolate one tracklet's site columns to t0_i, axis by axis."""
    tracklet_i = tracklet_rows(data, object_id, tracklet_id)
    times_i = tracklet_i[:, TIME_COLUMN]
    values_i = tracklet_i[:, columns]

    return np.array([
        np.interp(t0_i, times_i, values_i[:, axis])
        for axis in range(3)
    ])


def v_s_i_at_epoch(
    data,
    object_id,
    tracklet_id,
    t0_i,
):
    """Site velocity at t0_i, for Eq. (17)'s topocentric rates (stages 6-8)."""
    return _site_at_epoch(data, object_id, tracklet_id, t0_i, SITE_VELOCITY)


def build_range_bounds(variant):
    """Range bounds for every tracklet of one chain, written per chain."""
    data = load_data(variant)

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
        # [alpha, alpha_dot, delta, delta_dot]
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

    # Canonical.
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

    # For reading only.
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
    """Chains whose inputs exist; the others are skipped, not failed."""
    return [
        variant
        for variant in VARIANTS
        if data_file(variant).exists()
        and ATTRIBUTABLES_FILES[variant].exists()
    ]


def main():
    """Calculate range bounds for every chain."""
    for variant in VARIANTS:
        if variant not in available_variants():
            print(f"Skipping {variant}: inputs missing")
            continue

        print(f"--- {variant} ---")
        build_range_bounds(variant)
        print()


if __name__ == "__main__":
    main()
