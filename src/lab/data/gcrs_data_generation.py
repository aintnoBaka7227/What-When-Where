import argparse
import json

import numpy as np

from lab.data.dataset import (
    NOISY_VARIANTS,
    RELATIVE_NOISE_CASES_ARCSEC,
    data_file,
    settings_file,
)
from numba import jit
from scipy.spatial.transform import Rotation as R
from astropy.constants import GM_earth, R_earth
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    GCRS,
    ITRS,
    EarthLocation,
    CartesianRepresentation,
)


# Astropy constants are SI (m^3/s^2, m).
GM_EARTH = GM_earth.value
R_EARTH = R_earth.value

# km^3/s^2, the unit every orbit equation here uses.
mu = GM_EARTH / 1000000000


def gcrs_to_coe(gcrs):
    """Cartesian state [r, v] -> classical elements [rp, e, i, ohm, omega, v].

    rp is perigee radius, ohm the RAAN, omega the argument of perigee and
    v the true anomaly.
    """
    r = gcrs[0:3]
    v = gcrs[3:6]

    rmag = np.linalg.norm(r)
    vmag = np.linalg.norm(v)

    # Specific energy, then a from vis-viva.
    epsilon = (vmag * vmag / 2) - (mu / rmag)
    a = 1 / (-2 * epsilon / mu)

    rhat = r / rmag

    hvec = np.cross(r, v)
    hmag = np.linalg.norm(hvec)
    hhat = hvec / hmag

    # Eccentricity vector points to periapsis.
    ecc = np.cross(v, hvec) / mu - rhat
    e = np.linalg.norm(ecc)

    khat = np.array([0, 0, 1])
    i = np.arccos(np.dot(khat, hhat))

    ihat = np.array([1, 0, 0])
    nvec = np.cross(khat, hvec)
    nhat = nvec / np.linalg.norm(nvec)

    # The branch checks extend arccos's [0, pi] to [0, 2*pi).
    if (nhat[1] < 0):
        ohm = 2 * np.pi - np.arccos(np.dot(ihat, nhat))
    else:
        ohm = np.arccos(np.dot(ihat, nhat))

    if (ecc[2] < 0):
        omega = 2 * np.pi - np.arccos(np.dot(nhat, ecc / e))
    else:
        omega = np.arccos(np.dot(nhat, ecc / e))

    # r . v < 0 means moving inward: the second half of the orbit.
    if (np.dot(r, v) < 0):
        v = 2 * np.pi - np.arccos(np.dot(ecc / e, rhat))
    else:
        v = np.arccos(np.dot(ecc / e, rhat))

    rp = a * (1 - e)
    coe = np.array([rp, e, i, ohm, omega, v])

    return coe


def coe_to_gcrs(coe):
    """Classical elements [rp, e, i, ohm, omega, v] -> 6x1 state [r, v]."""
    rp = coe[0]  # km
    e = coe[1]
    i = coe[2]
    ohm = coe[3]
    omega = coe[4]
    v = coe[5]

    a = rp / (1 - e)
    p = a * (1 - e * e)

    # Position and velocity in the perifocal (PQW) frame.
    r_pqw_x = (p * np.cos(v)) / (1 + e * np.cos(v))
    r_pqw_y = (p * np.sin(v)) / (1 + e * np.cos(v))
    r_pqw = np.array([r_pqw_x, r_pqw_y, 0])

    v_pqw_x = - np.sqrt(mu / p) * np.sin(v)
    v_pqw_y = np.sqrt(mu / p) * (e + np.cos(v))

    v_pqw = np.array([v_pqw_x, v_pqw_y, 0])

    # PQW -> inertial rotation from RAAN, inclination and argument of perigee.
    rotm = np.zeros([3, 3])
    rotm[0, 0] = np.cos(ohm) * np.cos(omega) - np.sin(ohm) * np.sin(omega) * np.cos(i)
    rotm[0, 1] = -np.cos(ohm) * np.sin(omega) - np.sin(ohm) * np.cos(omega) * np.cos(i)
    rotm[0, 2] = np.sin(ohm) * np.sin(i)

    rotm[1, 0] = np.sin(ohm) * np.cos(omega) + np.cos(ohm) * np.sin(omega) * np.cos(i)
    rotm[1, 1] = -np.sin(ohm) * np.sin(omega) + np.cos(ohm) * np.cos(omega) * np.cos(i)
    rotm[1, 2] = -np.cos(ohm) * np.sin(i)

    rotm[2, 0] = np.sin(omega) * np.sin(i)
    rotm[2, 1] = np.cos(omega) * np.sin(i)
    rotm[2, 2] = np.cos(i)

    r = np.matmul(rotm, np.transpose(r_pqw))
    v = np.matmul(rotm, np.transpose(v_pqw))

    gcrs = np.zeros([6, 1])
    gcrs[0:3, 0] = r
    gcrs[3:6, 0] = v

    return gcrs


def oe_to_gcrs(oe):
    '''
    Equinoctial elements (a, ex, ey, hx, hy, l) -> state (px, py, pz, vx, vy, vz).

    Accepts arrays shaped (..., 6).
    '''
    a, ex, ey, hx, hy, l = oe.T.copy()

    p = a * (1 - (ex ** 2 + ey ** 2))

    s2 = 1 + hx ** 2 + hy ** 2
    alpha2 = hx ** 2 - hy ** 2

    r = p / (1 + ex * np.cos(l) + ey * np.sin(l))

    px = np.cos(l) + alpha2 * np.cos(l) + 2 * hx * hy * np.sin(l)
    py = np.sin(l) - alpha2 * np.sin(l) + 2 * hx * hy * np.cos(l)
    pz = 2 * (hx * np.sin(l) - hy * np.cos(l))

    pos = r / s2 * np.array([px, py, pz])

    vx = np.sin(l) + alpha2 * np.sin(l) - 2 * hx * hy * np.cos(l) + ey - 2 * ex * hx * hy + alpha2 * ey
    vy = -np.cos(l) + alpha2 * np.cos(l) + 2 * hx * hy * np.sin(l) - ex + 2 * ey * hx * hy + alpha2 * ex
    vz = -2 * (hx * np.cos(l) + hy * np.sin(l) + ex * hx + ey * hy)

    # km-based mu; the SI version was:
    # vel = -1 / s2 * np.sqrt(GM_EARTH / p) * np.array([vx, vy, vz])
    vel = -1 / s2 * np.sqrt(mu / p) * np.array([vx, vy, vz])

    gcrs = np.array([pos[0], pos[1], pos[2],
                     vel[0], vel[1], vel[2]]).T

    return gcrs


def gcrs_to_oe(gcrs):
    """
    State [r, v] -> equinoctial elements [a, f, g, h, k, L].

    Unused. Known bug: s2 = 1 + h + k should be 1 + h**2 + k**2.
    """
    r = gcrs[0:3]
    v = gcrs[3:6]

    # Only used by the commented-out vhat below.
    rdv = np.dot(r, v)

    rhat = r / np.linalg.norm(r)
    rmag = np.linalg.norm(r)

    hvec = np.cross(r, v)
    hmag = np.linalg.norm(hvec)
    hhat = hvec / hmag

    # vhat = (rmag * v - rdv * rhat) / hmag
    vhat = v / np.linalg.norm(v)
    vmag = np.linalg.norm(v)

    p = (hmag * hmag) / mu

    k = hhat[0] / (1 + hhat[2])
    h = -hhat[1] / (1 + hhat[2])

    kk = k * k
    hh = h * h
    s2 = 1 + h + k
    tkh = 2 * k * h

    ecc = np.cross(v, hvec) / mu - rhat

    # Equinoctial basis vectors.
    fhat = np.zeros([3])
    fhat[0] = 1 - kk + hh
    fhat[1] = tkh
    fhat[2] = -2 * k
    ghat = np.zeros([3])
    ghat[0] = tkh
    ghat[1] = 1 + kk - hh
    ghat[2] = 2 * h

    fhat = fhat / s2
    ghat = ghat / s2

    f = np.dot(ecc, fhat)
    g = np.dot(ecc, ghat)

    # Unused.
    X1 = np.dot(r, fhat)
    Y1 = np.dot(r, ghat)

    epsilon = (vmag * vmag / 2) - (mu / rmag)
    a = 1 / (-2 * epsilon / mu)

    L = np.arctan2(rhat[1] - vhat[0], rhat[0] + vhat[1])
    oe = np.array([a, f, g, h, k, L])

    return oe


def coe_to_eq_oe(coe):
    """
    Classical [rp, e, i, ohm, omega, v] -> equinoctial [a, ex, ey, hx, hy, lv].

    The propagator uses equinoctial elements, which have no singularity at
    e = 0 or i = 0.
    """
    rp = coe[0]  # km
    e = coe[1]
    i = coe[2]
    ohm = coe[3]
    omega = coe[4]
    v = coe[5]

    a = rp / (1 - e)

    ex = e * np.cos(omega + ohm)
    ey = e * np.sin(omega + ohm)

    hx = np.tan(i / 2) * np.cos(ohm)
    hy = np.tan(i / 2) * np.sin(ohm)

    # True longitude.
    lv = v + omega + ohm

    eq_oe = np.array([a, ex, ey, hx, hy, lv])
    return eq_oe


@jit(nopython=True, nogil=True, cache=True)
def two_body(mu, tau, ri, vi):
    """
    Propagate (ri, vi) by tau seconds under point-mass two-body motion.

    Universal-variable solve with a continued-fraction evaluation, valid for
    elliptic, parabolic and hyperbolic orbits. km, km/s, s.

    :return: rf, vf
    """
    tolerance = 1.0e-12

    # Universal variable, iterated until the implied time matches tau.
    u = np.float64(0.0)

    # Previous values, to detect a stalled iteration.
    uold = 100
    dtold = 100

    # imax = 20
    imax = 100

    # Explicit constants, since numba nopython mode cannot use sys.float_info.
    # umax = sys.float_info.max
    # umin = -sys.float_info.max
    umax = np.float64(1.7976931348623157e+308)
    umin = np.float64(-umax)

    orbits = 0

    tdesired = tau

    # Zero when tau = 0.
    threshold = tolerance * abs(tdesired)

    r0 = np.linalg.norm(ri)

    n0 = np.dot(ri, vi)

    # beta > 0 elliptic, = 0 parabolic, < 0 hyperbolic.
    beta = 2 * (mu / r0) - np.dot(vi, vi)

    if (beta != 0):
        umax = +1 / np.sqrt(abs(beta))
        umin = -1 / np.sqrt(abs(beta))

    # Complete revolutions contained in tau, for elliptic orbits.
    if (beta > 0):
        orbits = beta * tau - 2 * n0
        orbits = 1 + (orbits * np.sqrt(beta)) / (np.pi * mu)
        orbits = np.floor(orbits / 2)

    for i in range(imax):
        q = beta * u * u
        q = q / (1 + q)
        n = 0
        r = 1
        l = 1
        s = 1
        d = 3
        gcf = 1
        k = -5

        gold = 0

        # Continued fraction, evaluated until it stops changing.
        while (gcf != gold):
            k = -k
            l = l + 2
            d = d + 4 * l
            n = n + (1 + k) * l
            r = d / (d - n * r * q)
            s = (r - 1) * s
            gold = gcf
            gcf = gold + s

        h0 = 1 - 2 * q
        h1 = 2 * u * (1 - q)
        u0 = 2 * h0 * h0 - 1
        u1 = 2 * h0 * h1
        u2 = 2 * h1 * h1
        u3 = 2 * h1 * u2 * gcf / 3

        if (orbits != 0):
            u3 = u3 + 2 * np.pi * orbits / (beta * np.sqrt(beta))

        r1 = r0 * u0 + n0 * u1 + mu * u2
        dt = r0 * u1 + n0 * u2 + mu * u3

        # Newton slope d(dt)/du.
        slope = 4 * r1 / (1 + beta * u * u)

        terror = tdesired - dt

        if (abs(terror) < threshold):
            break

        if ((i > 1) and (u == uold)):
            break

        if ((i > 1) and (dt == dtold)):
            break

        uold = u
        dtold = dt

        # Newton step, bisecting if it leaves the bracket.
        ustep = terror / slope

        if (ustep > 0):
            umin = u
            u = u + ustep
            if (u > umax):
                u = (umin + umax) / 2
        else:
            umax = u
            u = u + ustep
            if (u < umin):
                u = (umin + umax) / 2

        # Never fires: i stops at imax - 1.
        if (i == imax):
            print('max iterations in twobody2 function')

    # Lagrange f and g coefficients.
    # usaved = u
    f = 1.0 - (mu / r0) * u2
    gg = 1.0 - (mu / r1) * u2
    g = r0 * u1 + n0 * u2
    ff = -mu * u1 / (r0 * r1)

    rf = f * ri + g * vi
    vf = ff * ri + gg * vi
    return rf, vf


@jit(nopython=True, nogil=True, cache=True)
def gcrs_to_radec(gcrs):
    '''
    Cartesian (px, py, pz[, vx, vy, vz]) -> (ra, dec), ra in [0, 2*pi).

    Reference: www.astrosurf.com/jephem/library/li110spherCart_en.htm
    '''
    px, py, pz = gcrs.T[:3]

    rho = np.sqrt(px ** 2 + py ** 2 + pz ** 2)

    ra = np.arctan2(py, px)

    if ra < 0:
        ra = 2 * np.pi + ra

    dec = np.arcsin(pz / rho)

    radec = np.array([ra, dec]).T

    return radec


def radec_to_point(radec):
    """(ra, dec) -> unit direction vector. Inverse of gcrs_to_radec; unused."""
    point = np.array([np.cos(radec[1]) * np.cos((radec[0])),
                      np.cos(radec[1]) * np.sin(radec[0]),
                      np.sin(radec[1])])

    return point

NEAR_GEO_ORBIT_RANGE_A = [38000, 45000]  # km
NEAR_GEO_INCLINATION_RANGE_DEG = [0.0, 20.0]  # deg

# Output row layout; lab.data.dataset names each column.
#   0-2   satellite position rx, ry, rz (GCRS)    km
#   3-5   satellite velocity vx, vy, vz (GCRS)    km/s
#   6-7   topocentric RA, Dec                     rad
#   8-10  station position (GCRS)                 km
#   11-13 station velocity (GCRS)                 km/s
#   14    time since epoch                        s
#   15    tracklet_id
#   16    object_id
#   17    observation_id
NUM_OF_OUTPUT_COLUMNS = 18
TRACKLET_ID_COLUMN = 15
OBJECT_ID_COLUMN = 16
OBSERVATION_ID_COLUMN = 17


def next_ids(output_path):
    """
    Next unused (tracklet_id, object_id, observation_id) for an output file.

    Files are appended to across runs, so IDs continue from the last row. A
    missing or empty file starts at zero.
    """
    try:
        with open(output_path, 'r') as f:
            lines = [line for line in f if line.strip()]
    except FileNotFoundError:
        return 0, 0, 0

    if not lines:
        return 0, 0, 0

    last_row = lines[-1].strip().split(',')
    if len(last_row) != NUM_OF_OUTPUT_COLUMNS:
        raise ValueError(
            f"{output_path} has {len(last_row)} columns per row, expected "
            f"{NUM_OF_OUTPUT_COLUMNS}. It was written by an older version of "
            f"this script with a different column layout; delete it and "
            f"regenerate."
        )

    return (
        int(last_row[TRACKLET_ID_COLUMN]) + 1,
        int(last_row[OBJECT_ID_COLUMN]) + 1,
        int(last_row[OBSERVATION_ID_COLUMN]) + 1,
    )


def load_settings_record(settings_path, settings, num_existing_objects):
    """
    Load the dataset's settings record, or start one.

    Raises before anything is written if the settings differ from those
    recorded, or if the record does not cover the objects already in the data.
    """
    try:
        with open(settings_path, 'r') as f:
            record = json.load(f)
    except FileNotFoundError:
        record = None

    if record is None:
        if num_existing_objects != 0:
            raise ValueError(
                f"{settings_path} is missing but the data files already hold "
                f"{num_existing_objects} object(s); delete all "
                f"generated_data_gcrs_* files and regenerate."
            )
        return {'settings': settings, 'objects': []}

    if record['settings'] != settings:
        changed = sorted(
            key for key in set(record['settings']) | set(settings)
            if record['settings'].get(key) != settings.get(key)
        )
        raise ValueError(
            f"Current settings differ from those recorded in {settings_path} "
            f"({', '.join(changed)}). Appending would mix datasets generated "
            f"with different settings; delete all generated_data_gcrs_* files "
            f"to start a new dataset, or restore the recorded settings."
        )

    if len(record['objects']) != num_existing_objects:
        raise ValueError(
            f"{settings_path} records {len(record['objects'])} object(s) but "
            f"the data files hold {num_existing_objects}; delete all "
            f"generated_data_gcrs_* files and regenerate."
        )

    return record


def fix_station_to_earth(site_gcrs_m, observation_time):
    """
    Attach a GCRS point (m) to the rotating Earth at observation_time.

    Converted once to ITRS, so the station stays on that patch of ground.
    """
    site_gcrs = GCRS(
        CartesianRepresentation(site_gcrs_m * u.m),
        obstime=observation_time,
    )
    site_itrs = site_gcrs.transform_to(ITRS(obstime=observation_time))

    return EarthLocation.from_geocentric(
        site_itrs.cartesian.x,
        site_itrs.cartesian.y,
        site_itrs.cartesian.z,
    )


def station_gcrs_state(station, observation_time):
    """
    A station's GCRS position (km) and velocity (km/s) at observation_time.

    Includes Earth rotation, precession, nutation and polar motion. The
    velocity is the r_s-dot term of Siminski et al. (2014) Eq. (6).
    """
    site_position, site_velocity = station.get_gcrs_posvel(observation_time)

    r_site = site_position.xyz.to_value(u.km)
    v_site = site_velocity.xyz.to_value(u.km / u.s)

    return r_site, v_site


def generate_object(test_case='a'):
    """
    Draw one object and append its rows to every data file.

    test_case 'a'-'d' picks the eccentricity range below. Seeded with the
    object ID, so object n is always identical.
    """
    # All cases share NEAR_GEO_ORBIT_RANGE_A.
    eccentricity_ranges = {
        'a': [0.0, 0.01],   # near-circular: the paper's near-GEO case
        'b': [0.01, 0.2],
        'c': [0.2, 0.4],
        'd': [0.4, 0.6],
    }
    a_min, a_max = NEAR_GEO_ORBIT_RANGE_A  # km

    settings_output = settings_file()
    settings_output.parent.mkdir(parents=True, exist_ok=True)

    # Appended to. Same orbits, stations and epochs; only RA/Dec differ.
    clean_output = data_file('clean')
    noisy_outputs = {
        sigma_arcsec: data_file(variant)
        for variant, sigma_arcsec in NOISY_VARIANTS.items()
    }

    # Siminski et al. (2014) Section 6.1 error model: relative noise per
    # observation, systematic bias once per tracklet (the ZimSMART epoch
    # offset). RA noise is not divided by cos(dec), matching fit_attributes.py.
    ARCSEC = np.deg2rad(1.0 / 3600.0)
    sigma_systematic = 5.0 * ARCSEC   # rad

    # IDs continue from existing rows; every file must be in step.
    first_tracklet_id, object_id, observation_id = next_ids(clean_output)
    for noisy_output in noisy_outputs.values():
        if next_ids(noisy_output) != (first_tracklet_id, object_id, observation_id):
            raise ValueError(
                f"{clean_output} and {noisy_output} are out of sync; "
                f"delete all generated_data_gcrs_* files and regenerate."
            )

    # The only source of randomness.
    rng = np.random.default_rng(object_id)

    # [rp, e, i, ohm, omega, v]. rp is drawn so that a is uniform on
    # [a_min, a_max] for the drawn e.
    if test_case == 'a': # nearly circular orbital test case
        e = rng.uniform(*eccentricity_ranges['a'])
        coe_oe = np.array([rng.uniform(a_min * (1 - e), a_max * (1 - e)),
                        e, #e
                        rng.uniform(*np.deg2rad(NEAR_GEO_INCLINATION_RANGE_DEG)), #i
                        rng.uniform(0, 2 * np.pi), #ohm
                        rng.uniform(0, 2 * np.pi), #omega
                        rng.uniform(0, 2 * np.pi)]) #v
    elif test_case == 'b':
        e = rng.uniform(*eccentricity_ranges['b'])
        coe_oe = np.array([rng.uniform(a_min * (1 - e), a_max * (1 - e)),
                        e,  # e
                        rng.uniform(*np.deg2rad(NEAR_GEO_INCLINATION_RANGE_DEG)),  # i
                        rng.uniform(0, 2 * np.pi),  # ohm
                        rng.uniform(0, 2 * np.pi),  # omega
                        rng.uniform(0, 2 * np.pi)])  # v
    elif test_case == 'c':
        e = rng.uniform(*eccentricity_ranges['c'])
        coe_oe = np.array([rng.uniform(a_min * (1 - e), a_max * (1 - e)),
                        e,  # e
                        rng.uniform(*np.deg2rad(NEAR_GEO_INCLINATION_RANGE_DEG)),  # i
                        rng.uniform(0, 2 * np.pi),  # ohm
                        rng.uniform(0, 2 * np.pi),  # omega
                        rng.uniform(0, 2 * np.pi)])  # v
    elif test_case == 'd':
        e = rng.uniform(*eccentricity_ranges['d'])
        coe_oe = np.array([rng.uniform(a_min * (1 - e), a_max * (1 - e)),
                        e,  # e
                        rng.uniform(*np.deg2rad(NEAR_GEO_INCLINATION_RANGE_DEG)),  # i
                        rng.uniform(0, 2 * np.pi),  # ohm
                        rng.uniform(0, 2 * np.pi),  # omega
                        rng.uniform(0, 2 * np.pi)])  # v


    # Irregular tracklet gaps, so pairs reach the degenerate 0/180 deg
    # transfer angles at different orbital periods rather than all at once.
    # Each object gets its own start offset and per-tracklet jitter.
    num_of_sites = 5

    tracklet_schedule_h = [0.0, 13.0, 29.0, 37.0, 53.0]

    max_start_offset_h = 24.0

    # Applied to every tracklet but the first, either way.
    max_tracklet_jitter_h = 2.0

    num_of_pts_in_each_site = 5

    # Table 2: ZimSMART tracklets last 1-2 min.
    measurement_time_in_each_site = 80 #seconds

    # 20 s spacing, matching ZimSMART.
    measurement_time_interval_within_each_site = measurement_time_in_each_site / (num_of_pts_in_each_site - 1)

    eq_oe = coe_to_eq_oe(coe_oe)
    rv_init = oe_to_gcrs(eq_oe)

    # Leading dimension, so rows index as rv_init[0, :].
    rv_init = rv_init[np.newaxis, :]

    # measurement_time = 0. First frame of Table 2.
    epoch = Time("2012-08-18T22:59:08", scale="utc")

    # Must match any existing record; load_settings_record enforces it.
    settings = {
        'frame': 'GCRS',
        'random_seed': 'object_id, via np.random.default_rng',
        'dynamics': 'two-body (point-mass Earth)',
        'epoch_utc': epoch.utc.isot,
        'mu_km3_s2': mu,
        'earth_radius_km': R_EARTH / 1000.0,
        'test_case': test_case,
        'semi_major_axis_range_km': [a_min, a_max],
        'eccentricity_range': eccentricity_ranges[test_case],
        'inclination_range_deg': NEAR_GEO_INCLINATION_RANGE_DEG,
        'station_max_offset_from_subsatellite_deg': 45.0,
        'num_of_tracklets_per_object': num_of_sites,
        'tracklet_schedule_h': tracklet_schedule_h,
        'max_start_offset_h': max_start_offset_h,
        'max_tracklet_jitter_h': max_tracklet_jitter_h,
        'num_of_observations_per_tracklet': num_of_pts_in_each_site,
        'tracklet_duration_s': measurement_time_in_each_site,
        'time_between_observations_s': measurement_time_interval_within_each_site,
        'relative_noise_levels_arcsec': RELATIVE_NOISE_CASES_ARCSEC,
        'systematic_noise_arcsec': sigma_systematic / ARCSEC,
        'noise_draws': 'unit normals shared across noise levels, scaled by sigma',
        'ra_noise_divided_by_cos_dec': False,
        'clean_file': clean_output.name,
        'noisy_files': {
            f'{sigma_arcsec}': path.name for sigma_arcsec, path in noisy_outputs.items()
        },
        'columns': [
            'rx_km', 'ry_km', 'rz_km',
            'vx_km_s', 'vy_km_s', 'vz_km_s',
            'ra_rad', 'dec_rad',
            'site_x_km', 'site_y_km', 'site_z_km',
            'site_vx_km_s', 'site_vy_km_s', 'site_vz_km_s',
            'time_s', 'tracklet_id', 'object_id', 'observation_id',
        ],
    }
    settings_record = load_settings_record(settings_output, settings, object_id)

    # This object's draws, recorded once its rows are written.
    object_record = {
        'object_id': object_id,
        'orbital_elements': {
            'perigee_radius_km': coe_oe[0],
            'semi_major_axis_km': eq_oe[0],
            'eccentricity': coe_oe[1],
            'inclination_deg': np.rad2deg(coe_oe[2]),
            'raan_deg': np.rad2deg(coe_oe[3]),
            'argument_of_perigee_deg': np.rad2deg(coe_oe[4]),
            'true_anomaly_deg': np.rad2deg(coe_oe[5]),
            'period_h': 2 * np.pi * np.sqrt(eq_oe[0] ** 3 / mu) / 3600.0,
        },
        'initial_state_km_km_s': rv_init[0].tolist(),
        'tracklets': [],
    }

    # Tracklet start times, s since the epoch.
    start_offset_h = rng.uniform(0.0, max_start_offset_h)
    jitter_h = rng.uniform(-max_tracklet_jitter_h, max_tracklet_jitter_h, num_of_sites)
    jitter_h[0] = 0.0
    tracklet_starts = (
        start_offset_h + np.array(tracklet_schedule_h) + jitter_h
    ) * 3600.0

    # One new station per tracklet.
    for i in range(num_of_sites):
        tracklet_id = first_tracklet_id + i

        # Per-tracklet bias, shared by every noisy file.
        ra_bias = rng.normal(0.0, sigma_systematic)
        dec_bias = rng.normal(0.0, sigma_systematic)

        for j in range(num_of_pts_in_each_site):
            measurement_time = (
                tracklet_starts[i]
                + j * measurement_time_interval_within_each_site
            )

            observation_time = epoch + measurement_time * u.s

            r, v = two_body(
                mu,
                measurement_time,
                rv_init[0, 0:3],
                rv_init[0, 3:6],
            )

            rv = np.zeros_like(rv_init)
            rv[0, 0:3] = r
            rv[0, 3:6] = v

            if j == 0: # setup the rsite
                # Start below the satellite.
                rsite = rv[0, 0:3] / np.linalg.norm(rv[0, 0:3])

                # A random axis perpendicular to rsite, so the rotation moves
                # the station by exactly rot_angle (up to 45 deg).
                rot_axis = np.array([
                    rng.uniform(-1, 1),
                    rng.uniform(-1, 1),
                    rng.uniform(-1, 1),
                ])
                rot_axis = rot_axis - np.dot(rot_axis, rsite) * rsite
                rot_axis = rot_axis / np.linalg.norm(rot_axis)
                rot_angle = rng.uniform(0, np.pi / 4)

                curr_R = R.from_rotvec(rot_angle * rot_axis)
                curr_R_mat = curr_R.as_matrix()

                # On the R_EARTH sphere, in m.
                rotated_rsite = np.matmul(curr_R_mat, rsite)
                rotated_rsite = rotated_rsite * R_EARTH

                # Fixed to the ground for the rest of the tracklet.
                station = fix_station_to_earth(rotated_rsite, observation_time)

                object_record['tracklets'].append({
                    'tracklet_id': tracklet_id,
                    'start_time_s': measurement_time,
                    'station_lat_deg': station.lat.deg,
                    'station_lon_deg': station.lon.deg,
                    'station_offset_from_subsatellite_deg': np.rad2deg(rot_angle),
                    'ra_systematic_bias_arcsec': ra_bias / ARCSEC,
                    'dec_systematic_bias_arcsec': dec_bias / ARCSEC,
                })

            r_site, v_site = station_gcrs_state(station, observation_time)

            # Geometric angles: no light-time, aberration or refraction.
            ra_true, dec_true = gcrs_to_radec(rv[0, 0:3] - r_site)

            # One unit draw per observation, scaled to each noise level, so
            # the noisy files differ only in amplitude.
            ra_unit_noise = rng.normal(0.0, 1.0)
            dec_unit_noise = rng.normal(0.0, 1.0)

            # Every file holds the true state; only RA/Dec differ.
            row_head = (
                f"{rv[0,0]},{rv[0,1]},{rv[0,2]},"
                f"{rv[0,3]},{rv[0,4]},{rv[0,5]},"
            )
            row_tail = (
                f"{r_site[0]},{r_site[1]},{r_site[2]},"
                f"{v_site[0]},{v_site[1]},{v_site[2]},"
                f"{measurement_time},"
                f"{tracklet_id},{object_id},{observation_id}\n"
            )

            with open(clean_output, 'a') as f:
                f.write(row_head + f"{ra_true},{dec_true}," + row_tail)

            for sigma_arcsec, noisy_output in noisy_outputs.items():
                sigma_relative = sigma_arcsec * ARCSEC  # rad
                ra_noisy = (
                    ra_true + ra_bias + sigma_relative * ra_unit_noise
                ) % (2.0 * np.pi)
                dec_noisy = dec_true + dec_bias + sigma_relative * dec_unit_noise

                with open(noisy_output, 'a') as f:
                    f.write(row_head + f"{ra_noisy},{dec_noisy}," + row_tail)

            observation_id += 1

    # Plain Python numbers, for JSON.
    def to_builtin(value):
        if isinstance(value, dict):
            return {key: to_builtin(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_builtin(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    settings_record['objects'].append(to_builtin(object_record))
    with open(settings_output, 'w') as f:
        json.dump(to_builtin(settings_record), f, indent=2)
        f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw one object and append it to the GCRS dataset.",
    )
    parser.add_argument("--test-case", default='a', choices=('a', 'b', 'c', 'd'))
    generate_object(parser.parse_args().test_case)
