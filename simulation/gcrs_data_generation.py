import json
from pathlib import Path

import numpy as np
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


# Astropy stores these constants in SI units:
#   GM_earth is in m^3/s^2
#   R_earth is in meters
GM_EARTH = GM_earth.value
R_EARTH = R_earth.value

# The orbital equations below use kilometers for satellite position, so convert
# Earth's gravitational parameter from m^3/s^2 to km^3/s^2.
mu = GM_EARTH / 1000000000


def gcrs_to_coe(gcrs):
    """
    Convert a Cartesian inertial state into classical orbital elements.

    Input gcrs:
        [rx, ry, rz, vx, vy, vz]

    Output coe:
        [rp, e, i, ohm, omega, v]
        rp    = perigee radius
        e     = eccentricity
        i     = inclination
        ohm   = right ascension of ascending node
        omega = argument of perigee
        v     = true anomaly
    """
    # Split the 6-element state vector into position and velocity.
    r = gcrs[0:3]
    v = gcrs[3:6]

    # Magnitudes of position and velocity are used in energy/orbit formulas.
    rmag = np.linalg.norm(r)
    vmag = np.linalg.norm(v)

    # Specific orbital energy. For a bound elliptical orbit, epsilon is negative.
    epsilon = (vmag * vmag / 2) - (mu / rmag)

    # Semi-major axis from specific orbital energy.
    a = 1 / (-2 * epsilon / mu)

    # Unit position vector.
    rhat = r / rmag

    # Specific angular momentum vector. Its direction is normal to the orbit plane.
    hvec = np.cross(r, v)
    hmag = np.linalg.norm(hvec)
    hhat = hvec / hmag

    # Eccentricity vector points toward periapsis; its magnitude is eccentricity.
    ecc = np.cross(v, hvec) / mu - rhat
    e = np.linalg.norm(ecc)

    # Inclination is the angle between the orbit normal and Earth's z-axis.
    khat = np.array([0, 0, 1])
    i = np.arccos(np.dot(khat, hhat))

    # Node vector points toward the ascending node, where the orbit crosses the
    # equatorial plane heading north.
    ihat = np.array([1, 0, 0])
    nvec = np.cross(khat, hvec)
    nhat = nvec / np.linalg.norm(nvec)

    # Right ascension of ascending node. The branch check keeps the angle in
    # the full 0 to 2*pi range instead of only 0 to pi.
    if (nhat[1] < 0):
        ohm = 2 * np.pi - np.arccos(np.dot(ihat, nhat))
    else:
        ohm = np.arccos(np.dot(ihat, nhat))

    # Argument of perigee: angle from ascending node to eccentricity vector.
    if (ecc[2] < 0):
        omega = 2 * np.pi - np.arccos(np.dot(nhat, ecc / e))
    else:
        omega = np.arccos(np.dot(nhat, ecc / e))

    # True anomaly: angle from perigee to the current position.
    # If r dot v is negative, the satellite is moving inward, so use the
    # second-half-of-orbit angle.
    if (np.dot(r, v) < 0):
        v = 2 * np.pi - np.arccos(np.dot(ecc / e, rhat))
    else:
        v = np.arccos(np.dot(ecc / e, rhat))

    # Perigee radius is the closest orbital radius: a * (1 - e).
    rp = a * (1 - e)
    coe = np.array([rp, e, i, ohm, omega, v])

    return coe


def coe_to_gcrs(coe):
    """
    Convert classical orbital elements into Cartesian position and velocity.

    Input coe:
        [rp, e, i, ohm, omega, v]

    Output gcrs:
        6x1 vector [rx, ry, rz, vx, vy, vz]
    """
    # Unpack orbital elements. Angles are in radians.
    rp = coe[0]  # takes km
    e = coe[1]
    i = coe[2]
    ohm = coe[3]
    omega = coe[4]
    v = coe[5]

    # Convert perigee radius to semi-major axis, then compute semi-latus rectum.
    a = rp / (1 - e)
    p = a * (1 - e * e)

    # Position in the perifocal PQW frame, where the orbit lies in the x-y plane.
    r_pqw_x = (p * np.cos(v)) / (1 + e * np.cos(v))
    r_pqw_y = (p * np.sin(v)) / (1 + e * np.cos(v))
    r_pqw = np.array([r_pqw_x, r_pqw_y, 0])

    # Velocity in the same perifocal frame.
    v_pqw_x = - np.sqrt(mu / p) * np.sin(v)
    v_pqw_y = np.sqrt(mu / p) * (e + np.cos(v))

    v_pqw = np.array([v_pqw_x, v_pqw_y, 0])

    # Rotation matrix from perifocal coordinates into the inertial GCRS-like frame.
    # It combines RAAN (ohm), inclination (i), and argument of perigee (omega).
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

    # Rotate position and velocity out of the orbital plane.
    r = np.matmul(rotm, np.transpose(r_pqw))
    v = np.matmul(rotm, np.transpose(v_pqw))

    # Pack position and velocity into a 6x1 state vector.
    gcrs = np.zeros([6, 1])
    gcrs[0:3, 0] = r
    gcrs[3:6, 0] = v

    return gcrs


def oe_to_gcrs(oe):
    '''
    Converts orbital elements into position and velocity.

    Parameters
    ----------
    oe : np.array (...,6)
        Orbital elements (a,ex,ey,hx,hy,l).

    Returns
    -------
    gcrs : np.array (...,6)
        Position and velocity (px,py,pz,vx,vy,vz).
    '''
    # Assign the equinoctial elements. The .T.copy() form lets this function
    # work with arrays shaped like (..., 6), not just one flat 6-item vector.
    a, ex, ey, hx, hy, l = oe.T.copy()

    # Semi-latus rectum. In equinoctial form, eccentricity is split across ex/ey.
    p = a * (1 - (ex ** 2 + ey ** 2))

    # Repeated terms used by the equinoctial position/velocity equations.
    s2 = 1 + hx ** 2 + hy ** 2
    alpha2 = hx ** 2 - hy ** 2

    # Orbital radius at longitude l.
    r = p / (1 + ex * np.cos(l) + ey * np.sin(l))

    # Direction terms for the inertial position vector.
    px = np.cos(l) + alpha2 * np.cos(l) + 2 * hx * hy * np.sin(l)
    py = np.sin(l) - alpha2 * np.sin(l) + 2 * hx * hy * np.cos(l)
    pz = 2 * (hx * np.sin(l) - hy * np.cos(l))

    # Scale the direction by orbital radius.
    pos = r / s2 * np.array([px, py, pz])

    # Direction terms for the inertial velocity vector.
    vx = np.sin(l) + alpha2 * np.sin(l) - 2 * hx * hy * np.cos(l) + ey - 2 * ex * hx * hy + alpha2 * ey
    vy = -np.cos(l) + alpha2 * np.cos(l) + 2 * hx * hy * np.sin(l) - ex + 2 * ey * hx * hy + alpha2 * ex
    vz = -2 * (hx * np.cos(l) + hy * np.sin(l) + ex * hx + ey * hy)

    # Scale velocity by sqrt(mu / p). The commented line above used SI mu;
    # this version uses the km-based mu that matches the rest of the script.
    # vel = -1 / s2 * np.sqrt(GM_EARTH / p) * np.array([vx, vy, vz])
    vel = -1 / s2 * np.sqrt(mu / p) * np.array([vx, vy, vz])

    # Construct the position/velocity state matrix:
    # [rx, ry, rz, vx, vy, vz].
    gcrs = np.array([pos[0], pos[1], pos[2],
                     vel[0], vel[1], vel[2]]).T

    return gcrs


def gcrs_to_oe(gcrs):
    """
    Convert a Cartesian inertial state back into equinoctial orbital elements.

    Output:
        [a, f, g, h, k, L]

    The current main script does not call this function, but it is the inverse
    companion to oe_to_gcrs for many orbital-element workflows.
    """
    # Split the incoming state into position and velocity.
    r = gcrs[0:3]
    v = gcrs[3:6]

    # Dot product between position and velocity. This is computed here but is
    # only used by the commented-out vhat formula below.
    rdv = np.dot(r, v)

    # Position direction and magnitude.
    rhat = r / np.linalg.norm(r)
    rmag = np.linalg.norm(r)

    # Angular momentum vector defines the orbit plane.
    hvec = np.cross(r, v)
    hmag = np.linalg.norm(hvec)
    hhat = hvec / hmag

    # Velocity direction and magnitude.
    # vhat = (rmag * v - rdv * rhat) / hmag
    vhat = v / np.linalg.norm(v)
    vmag = np.linalg.norm(v)

    # Semi-latus rectum from angular momentum.
    p = (hmag * hmag) / mu

    # Equinoctial inclination components derived from angular momentum direction.
    k = hhat[0] / (1 + hhat[2])
    h = -hhat[1] / (1 + hhat[2])

    # Repeated terms for the fhat/ghat basis vectors.
    kk = k * k
    hh = h * h
    s2 = 1 + h + k
    tkh = 2 * k * h

    # Eccentricity vector, projected later onto the equinoctial basis.
    ecc = np.cross(v, hvec) / mu - rhat

    # fhat and ghat are basis vectors used to express eccentricity components.
    fhat = np.zeros([3])
    fhat[0] = 1 - kk + hh
    fhat[1] = tkh
    fhat[2] = -2 * k
    ghat = np.zeros([3])
    ghat[0] = tkh
    ghat[1] = 1 + kk - hh
    ghat[2] = 2 * h

    # Normalize the basis vectors.
    fhat = fhat / s2
    ghat = ghat / s2

    # Equinoctial eccentricity components.
    f = np.dot(ecc, fhat)
    g = np.dot(ecc, ghat)

    # Coordinates of r in the fhat/ghat frame. These are calculated but not used
    # by the final expression below.
    X1 = np.dot(r, fhat)
    Y1 = np.dot(r, ghat)

    # Semi-major axis from specific orbital energy.
    epsilon = (vmag * vmag / 2) - (mu / rmag)
    a = 1 / (-2 * epsilon / mu)

    # Longitude-like angle for the equinoctial representation.
    L = np.arctan2(rhat[1] - vhat[0], rhat[0] + vhat[1])
    oe = np.array([a, f, g, h, k, L])

    return oe


def coe_to_eq_oe(coe):
    """
    Convert classical orbital elements to equinoctial orbital elements.
    equinoctial elements are used by the propagator because they avoid singularities
    at zero eccentricity(circular orbit e = 0) and zero inclination(i=0).

    Classical input:
        [rp, e, i, ohm, omega, v]

    Equinoctial output:
        [a, ex, ey, hx, hy, lv]
    """
    # Unpack the classical elements.
    rp = coe[0]  # takes km
    e = coe[1]
    i = coe[2]
    ohm = coe[3]
    omega = coe[4]
    v = coe[5]

    # Convert perigee radius to semi-major axis.
    a = rp / (1 - e)

    # Store eccentricity as x/y components instead of magnitude + angle.
    ex = e * np.cos(omega + ohm)
    ey = e * np.sin(omega + ohm)

    # Store inclination/RAAN as x/y components using tan(i / 2).
    hx = np.tan(i / 2) * np.cos(ohm)
    hy = np.tan(i / 2) * np.sin(ohm)

    # True longitude is true anomaly plus argument of perigee plus RAAN.
    lv = v + omega + ohm

    eq_oe = np.array([a, ex, ey, hx, hy, lv])
    return eq_oe


@jit(nopython=True, nogil=True, cache=True)
def two_body(mu, tau, ri, vi):
    """
    Propagate a satellite under ideal two-body motion.

    This ignores drag, Earth's oblateness, third-body gravity, maneuvers, and
    measurement noise. It only models motion caused by a point-mass Earth.

    :param mu: gravitational constant(km ** 3 / sec ** 2)
    :param tau: propagation time interval(seconds)
    :param ri: initial eci position vector(kilometers)
    :param vi: initial eci velocity vector(kilometers / second)
    :return:
    rf = final eci position vector(kilometers)
    vf = final eci velocity vector(kilometers / second)
    """
    # Numerical tolerance for solving the universal-variable time equation.
    tolerance = 1.0e-12

    # Universal anomaly / universal variable. The loop below adjusts this until
    # the computed propagation time matches tau.
    u = np.float64(0.0)

    # Previous iteration values used to detect stalled convergence.
    uold = 100
    dtold = 100

    # Maximum number of iterations for the universal-variable solve.
    # imax = 20
    imax = 100

    # Initial search bounds for u. These replace sys.float_info because Numba's
    # nopython mode works best with explicit numeric constants.
    # umax = sys.float_info.max
    # umin = -sys.float_info.max
    umax = np.float64(1.7976931348623157e+308)
    umin = np.float64(-umax)

    # Number of complete revolutions included in the propagation interval.
    orbits = 0

    # Desired elapsed time.
    tdesired = tau

    # Absolute time-error threshold. For tau = 0, this threshold is also 0.
    threshold = tolerance * abs(tdesired)

    # Initial orbital radius and radial velocity-like dot product.
    r0 = np.linalg.norm(ri)

    n0 = np.dot(ri, vi)

    # Energy-related parameter. Positive beta corresponds to elliptic motion,
    # zero to parabolic, and negative to hyperbolic.
    beta = 2 * (mu / r0) - np.dot(vi, vi)

    # For non-parabolic motion, use beta to set a tighter valid interval for u.
    if (beta != 0):
        umax = +1 / np.sqrt(abs(beta))
        umin = -1 / np.sqrt(abs(beta))

    # Elliptic orbits can wrap around the central body multiple times during tau.
    # This estimates how many full revolutions must be included.
    if (beta > 0):
        orbits = beta * tau - 2 * n0
        orbits = 1 + (orbits * np.sqrt(beta)) / (np.pi * mu)
        orbits = np.floor(orbits / 2)

    # Solve for the universal variable u by iteration.
    for i in range(imax):
        # q and the continued fraction terms are part of a universal-variable
        # formulation that works across elliptic, parabolic, and hyperbolic cases.
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

        # Evaluate the continued fraction until it stops changing.
        while (gcf != gold):
            k = -k
            l = l + 2
            d = d + 4 * l
            n = n + (1 + k) * l
            r = d / (d - n * r * q)
            s = (r - 1) * s
            gold = gcf
            gcf = gold + s

        # Universal functions used to compute time and the Lagrange coefficients.
        h0 = 1 - 2 * q
        h1 = 2 * u * (1 - q)
        u0 = 2 * h0 * h0 - 1
        u1 = 2 * h0 * h1
        u2 = 2 * h1 * h1
        u3 = 2 * h1 * u2 * gcf / 3

        # Add complete-orbit time contribution for elliptic multi-revolution cases.
        if (orbits != 0):
            u3 = u3 + 2 * np.pi * orbits / (beta * np.sqrt(beta))

        # r1 is the final radius implied by the current u.
        r1 = r0 * u0 + n0 * u1 + mu * u2

        # dt is the propagation time implied by the current u.
        dt = r0 * u1 + n0 * u2 + mu * u3

        # Derivative-like slope used to update u with a Newton-style step.
        slope = 4 * r1 / (1 + beta * u * u)

        # Remaining time error.
        terror = tdesired - dt

        # Stop if the computed time is close enough to the requested time.
        if (abs(terror) < threshold):
            break

        # Stop if u or dt is no longer changing, which prevents an infinite loop.
        if ((i > 1) and (u == uold)):
            break

        if ((i > 1) and (dt == dtold)):
            break

        uold = u
        dtold = dt

        # Proposed update step for u.
        ustep = terror / slope

        # Update lower/upper bounds and keep u inside the current bracket.
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

        if (i == imax):
            print('max iterations in twobody2 function')

    # Lagrange f and g coefficients map the initial state to the final state.
    # usaved = u
    f = 1.0 - (mu / r0) * u2
    gg = 1.0 - (mu / r1) * u2
    g = r0 * u1 + n0 * u2
    ff = -mu * u1 / (r0 * r1)

    # Final position and velocity after tau seconds.
    rf = f * ri + g * vi
    vf = ff * ri + gg * vi
    return rf, vf







@jit(nopython=True, nogil=True, cache=True)
def gcrs_to_radec(gcrs):
    '''
    Converts cartesian gcrs into pointing coordinates (eci_frame <-> ra/dec).
    Loose reference: www.astrosurf.com/jephem/library/li110spherCart_en.htm

    Parameters
    ----------
    gcrs : np.array (...,3) or (...,6)
        Cartesian coordinates (px,py,pz,(vx,vy,vz)).

    Returns
    -------
    radec : np.array (...,2)
        Spherical coordinates (ra, dec).
    '''
    # Separate and assign the Cartesian position components. If a 6-element
    # state is passed in, only the first 3 position values are used.
    px, py, pz = gcrs.T[:3]

    # Distance from the origin to the point.
    rho = np.sqrt(px ** 2 + py ** 2 + pz ** 2)

    # Right ascension is the angle in the x-y plane measured from +x toward +y.
    ra = np.arctan2(py, px)

    # arctan2 returns values from -pi to pi; convert negative values into
    # the 0 to 2*pi range used by many astronomy workflows.
    if ra < 0:
        ra = 2 * np.pi + ra

    # Declination is the angle above/below the x-y plane.
    dec = np.arcsin(pz / rho)

    radec = np.array([ra, dec]).T

    return radec




def radec_to_point(radec):
    """
    Convert right ascension and declination into a 3D unit direction vector.

    This helper is not used in the main generation block, but it is the inverse
    idea of gcrs_to_radec: angles in, unit pointing vector out.
    """
    # Standard spherical-to-Cartesian conversion for a unit vector.
    point = np.array([np.cos(radec[1]) * np.cos((radec[0])),
                      np.cos(radec[1]) * np.sin(radec[0]),
                      np.sin(radec[1])])

    return point

NEAR_GEO_ORBIT_RANGE_A = [38000, 45000]  # km

# Relative angular noise levels tested by Siminski et al. (2014), Section 6.1:
# sigma_alpha,delta in {0.7", 1.2", 2"}. Every run writes one noisy file per level.
RELATIVE_NOISE_CASES_ARCSEC = [0.7, 1.2, 2.0]  # arcsec

# Output row layout. Columns 0-16 match new_data_gen.py's dataset layout;
# column 17 is an extra per-observation ID.
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
    Return the next unused (tracklet_id, object_id, observation_id) for an
    output file.

    The output file is appended to across runs, so IDs continue from the last
    row already written rather than restarting at zero. A missing or empty file
    starts all IDs at zero.
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
    Load the settings record that accompanies the data files, or start one.

    The record holds the run-independent generation settings once, plus one
    entry per generated object. Because the data files are appended to across
    runs, every run must use the same settings; this raises before any data is
    written if the current settings differ from the recorded ones, or if the
    record does not cover the objects already in the data files.
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
    Attach a GCRS point to the rotating Earth.

    site_gcrs_m is a station position along GCRS axes, in meters, valid at
    observation_time. It is converted once to ITRS, the Earth-fixed frame, so
    the returned EarthLocation is the patch of ground under that point and
    rotates with Earth from then on.
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
    Return a ground station's GCRS position (km) and velocity (km/s).

    Astropy rotates the station's fixed ITRS coordinates into GCRS at
    observation_time, including Earth rotation, precession, nutation and polar
    motion. The velocity is the station's motion as Earth carries it around,
    the r_s-dot term of Siminski et al. (2014) Eq. (6).
    """
    site_position, site_velocity = station.get_gcrs_posvel(observation_time)

    r_site = site_position.xyz.to_value(u.km)
    v_site = site_velocity.xyz.to_value(u.km / u.s)

    return r_site, v_site


if __name__ == "__main__":
    # Choose which random orbit family to generate. Change this to 'b', 'c',
    # or 'd' to use the other eccentricity/perigee-radius ranges below.
    test_case = 'a'

    # Eccentricity range of each test case. All cases share the semi-major axis
    # range NEAR_GEO_ORBIT_RANGE_A.
    eccentricity_ranges = {
        'a': [0.0, 0.01],   # near-circular: the paper's near-GEO case
        'b': [0.01, 0.2],
        'c': [0.2, 0.4],
        'd': [0.4, 0.6],
    }
    a_min, a_max = NEAR_GEO_ORBIT_RANGE_A  # km

    # Settings record: the run-independent settings plus every object's drawn
    # orbit, stations and biases, so a dataset can be described and audited.
    # All outputs go to the project's data/ folder, resolved from this file's
    # location so the working directory does not matter.
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    settings_output = data_dir / 'generated_data_gcrs_settings.json'

    # Output files. The script opens these in append mode later, so repeated
    # runs add rows instead of replacing old rows. All files are written from
    # the same orbit, stations and epochs and differ only in the RA/Dec columns:
    # the clean file holds the true angles, each noisy file adds measurement
    # error at one relative noise level, named in its filename.
    clean_output = data_dir / 'generated_data_gcrs_clean.txt'
    noisy_outputs = {
        sigma_arcsec: data_dir / f'generated_data_gcrs_noisy_{sigma_arcsec}as.txt'
        for sigma_arcsec in RELATIVE_NOISE_CASES_ARCSEC
    }

    # Siminski et al. (2014) measurement-error model (Section 6.1).
    # Relative noise is redrawn for every observation; the systematic error is
    # drawn once per tracklet and held constant, modelling the ZimSMART
    # exposure-epoch offset. Set sigma_systematic to 0 for relative noise only.
    # RA noise is not divided by cos(dec), matching new_data_gen.py.
    ARCSEC = np.deg2rad(1.0 / 3600.0)
    sigma_systematic = 5.0 * ARCSEC   # rad

    # Each test case builds one random set of classical orbital elements:
    #   [rp, e, i, ohm, omega, v]
    # where rp is perigee radius, e is eccentricity, i is inclination,
    # ohm is RAAN, omega is argument of perigee, and v is true anomaly.
    if test_case == 'a': # nearly circular orbital test case
        # Eccentricity 0-0.01; rp chosen so semi-major axis is 38000-45000 km.
        e = np.random.uniform(*eccentricity_ranges['a'])
        coe_oe = np.array([np.random.uniform(a_min * (1 - e), a_max * (1 - e)),
                        e, #e
                        np.random.uniform(np.deg2rad(1), np.deg2rad(179)), #i
                        np.random.uniform(0, 2 * np.pi), #ohm
                        np.random.uniform(0, 2 * np.pi), #omega
                        np.random.uniform(0, 2 * np.pi)]) #v
    elif test_case == 'b':
        # Eccentricity 0.01-0.2; rp chosen so semi-major axis is 38000-45000 km.
        e = np.random.uniform(*eccentricity_ranges['b'])
        coe_oe = np.array([np.random.uniform(a_min * (1 - e), a_max * (1 - e)),
                        e,  # e
                        np.random.uniform(np.deg2rad(1), np.deg2rad(179)),  # i
                        np.random.uniform(0, 2 * np.pi),  # ohm
                        np.random.uniform(0, 2 * np.pi),  # omega
                        np.random.uniform(0, 2 * np.pi)])  # v
    elif test_case == 'c':
        # Eccentricity 0.2-0.4; rp chosen so semi-major axis is 38000-45000 km.
        e = np.random.uniform(*eccentricity_ranges['c'])
        coe_oe = np.array([np.random.uniform(a_min * (1 - e), a_max * (1 - e)),
                        e,  # e
                        np.random.uniform(np.deg2rad(1), np.deg2rad(179)),  # i
                        np.random.uniform(0, 2 * np.pi),  # ohm
                        np.random.uniform(0, 2 * np.pi),  # omega
                        np.random.uniform(0, 2 * np.pi)])  # v
    elif test_case == 'd':
        # Eccentricity 0.4-0.6; rp chosen so semi-major axis is 38000-45000 km.
        e = np.random.uniform(*eccentricity_ranges['d'])
        coe_oe = np.array([np.random.uniform(a_min * (1 - e), a_max * (1 - e)),
                        e,  # e
                        np.random.uniform(np.deg2rad(1), np.deg2rad(179)),  # i
                        np.random.uniform(0, 2 * np.pi),  # ohm
                        np.random.uniform(0, 2 * np.pi),  # omega
                        np.random.uniform(0, 2 * np.pi)])  # v


    # Data-set size and timing controls.
    # Five sites 13 h apart span 0-52 h, so tracklet pairs are separated by
    # 13, 26, 39 and 52 h: k = 0.5, 1, 1.5 and 2 revolutions for a GEO object.
    # 13 h rather than 12 h because multiples of 12 h are almost exactly
    # multiples of half a GEO period, giving ~0 or ~180 deg transfer angles
    # where Lambert's problem is degenerate (r1 and r2 nearly collinear).
    num_of_sites = 5

    # Seconds between the start of one site observation window and the next.
    time_interval_between_sites = 13 * 3600 #seconds

    # Number of measurements generated for each site.
    num_of_pts_in_each_site = 5

    # Tracklet duration: time from the first to the last measurement of a site.
    # Siminski et al. (2014) Table 2 gives 1-2 min for the ZimSMART tracklets.
    measurement_time_in_each_site = 80 #seconds

    # Time spacing between measurements within a site. With 80 seconds and
    # 5 points, the script samples at 0, 20, 40, 60, and 80 seconds, matching
    # the ~20 s spacing of the ZimSMART tracklets in Siminski et al. (2014).
    measurement_time_interval_within_each_site = measurement_time_in_each_site / (num_of_pts_in_each_site - 1)

    # Convert classical orbital elements to equinoctial elements, then to the
    # initial Cartesian position/velocity state used by the propagator.
    eq_oe = coe_to_eq_oe(coe_oe)
    rv_init = oe_to_gcrs(eq_oe)

    # Add a leading dimension so the rest of the code can index as rv_init[0, :].
    rv_init = rv_init[np.newaxis, :]

    # Absolute epoch corresponding to measurement_time = 0 seconds.
    # Adopt GCRS axes for the synthetic satellite position and velocity.
    epoch = Time("2012-08-18T22:59:08", scale="utc")

    # Unique IDs, continued from any rows already in the output files:
    # one object_id per run, one tracklet_id per site window, one
    # observation_id per row. The clean and noisy files share IDs row for row,
    # so they must all be in step.
    first_tracklet_id, object_id, observation_id = next_ids(clean_output)
    for noisy_output in noisy_outputs.values():
        if next_ids(noisy_output) != (first_tracklet_id, object_id, observation_id):
            raise ValueError(
                f"{clean_output} and {noisy_output} are out of sync; "
                f"delete all generated_data_gcrs_* files and regenerate."
            )

    # Run-independent settings. Every run appending to the same data files
    # must use exactly these, which load_settings_record enforces.
    settings = {
        'frame': 'GCRS',
        'dynamics': 'two-body (point-mass Earth)',
        'epoch_utc': epoch.utc.isot,
        'mu_km3_s2': mu,
        'earth_radius_km': R_EARTH / 1000.0,
        'test_case': test_case,
        'semi_major_axis_range_km': [a_min, a_max],
        'eccentricity_range': eccentricity_ranges[test_case],
        'inclination_range_deg': [1.0, 179.0],
        'station_max_offset_from_subsatellite_deg': 45.0,
        'num_of_tracklets_per_object': num_of_sites,
        'time_between_tracklet_starts_s': time_interval_between_sites,
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

    # This object's randomly drawn values, added to the settings record after
    # its rows are written.
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

    # Outer loop: create one independent observing station per tracklet.
    for i in range(num_of_sites):
        tracklet_id = first_tracklet_id + i

        # One constant RA bias and one constant Dec bias for this tracklet,
        # applied identically to every noisy file.
        ra_bias = np.random.normal(0.0, sigma_systematic)
        dec_bias = np.random.normal(0.0, sigma_systematic)

        # Inner loop: observations from the same station.
        for j in range(num_of_pts_in_each_site):
            # Elapsed propagation time in seconds.
            measurement_time = (
                i * time_interval_between_sites
                + j * measurement_time_interval_within_each_site
            )

            # Absolute timestamp used by Astropy.
            observation_time = epoch + measurement_time * u.s

            # Propagate the satellite to the current observation time.
            # Position: km. Velocity: km/s.
            r, v = two_body(
                mu,
                measurement_time,
                rv_init[0, 0:3],
                rv_init[0, 3:6],
            )

            # Pack propagated position and velocity into the original shape.
            rv = np.zeros_like(rv_init)
            rv[0, 0:3] = r
            rv[0, 3:6] = v

            if j == 0: # setup the rsite
                # Start with the satellite radial direction at the first point
                # in this site window.
                rsite = rv[0, 0:3] / np.linalg.norm(rv[0, 0:3])

                # Pick a random axis, normalize it, then rotate the radial
                # direction by up to 45 degrees.
                rot_axis = np.array([
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1),
                ])
                rot_axis = rot_axis / np.linalg.norm(rot_axis)
                rot_angle = np.random.uniform(0, np.pi / 4)

                # scipy Rotation expects a rotation vector:
                # direction = rotation axis, magnitude = rotation angle.
                curr_R = R.from_rotvec(rot_angle * rot_axis)
                curr_R_mat = curr_R.as_matrix()

                # Initial station position on the spherical Earth, in meters.
                rotated_rsite = np.matmul(curr_R_mat, rsite)
                rotated_rsite = rotated_rsite * R_EARTH

                # Attach the chosen GCRS point to the ground at this tracklet's
                # first observation time, and keep that station for all
                # observations in this tracklet.
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

            # For every observation, calculate this station's current GCRS
            # position (km) and velocity (km/s) as Earth carries it around.
            r_site, v_site = station_gcrs_state(station, observation_time)

            # Geometric topocentric RA/Dec from the updated station position.
            ra_true, dec_true = gcrs_to_radec(rv[0, 0:3] - r_site)

            # Fresh per-observation noise, drawn once as a unit normal and then
            # scaled to each noise level. The noisy files therefore share the
            # same underlying draws and differ only in noise amplitude, so a
            # change in results between levels is due to sigma, not to luck.
            ra_unit_noise = np.random.normal(0.0, 1.0)
            dec_unit_noise = np.random.normal(0.0, 1.0)

            # Row layout is documented at NUM_OF_OUTPUT_COLUMNS. RA/Dec sit
            # between row_head and row_tail. The satellite state is the true
            # state in every file; only the measured angles differ.
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

            # Noisy angles: per-tracklet bias plus scaled per-observation noise.
            for sigma_arcsec, noisy_output in noisy_outputs.items():
                sigma_relative = sigma_arcsec * ARCSEC  # rad
                ra_noisy = (
                    ra_true + ra_bias + sigma_relative * ra_unit_noise
                ) % (2.0 * np.pi)
                dec_noisy = dec_true + dec_bias + sigma_relative * dec_unit_noise

                with open(noisy_output, 'a') as f:
                    f.write(row_head + f"{ra_noisy},{dec_noisy}," + row_tail)

            observation_id += 1

    # Record this object and rewrite the settings file. Values are converted
    # to plain floats so the JSON holds ordinary numbers.
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
                
