from pathlib import Path

import numpy as np
from numba import jit
from scipy.spatial.transform import Rotation as R
from astropy.constants import GM_earth, R_earth
import astropy.units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
from astropy.time import Time
from astropy.utils import iers


iers.conf.auto_download = True
iers.conf.auto_max_age = None


# Astropy stores these constants in SI units:
#   GM_earth is in m^3/s^2
#   R_earth is in meters
GM_EARTH = GM_earth.value

# The orbital equations below use kilometers for satellite position, so convert
# Earth's gravitational parameter from m^3/s^2 to km^3/s^2.
mu = GM_EARTH / 1000000000

# Earth's equatorial radius in kilometers, used to turn a pointing direction
# into a position on the globe.
R_EARTH_KM = R_earth.value / 1000


# Orbit population each test case samples from.
#
# hypothesis/range_bounds.py must bound the same population, or the true
# range falls outside the region the optimizer is allowed to search and
# every association silently fails. These limits are therefore written into
# the generated .npz so the bounds can be derived from the data rather than
# hardcoded to whichever case was current.
SEMI_MAJOR_AXIS_LIMITS_KM = (38_000.0, 45_000.0)

ECCENTRICITY_LIMITS = {
    'a': (0.0, 0.01),    # near-circular: the paper's near-GEO test case
    'b': (0.01, 0.2),
    'c': (0.2, 0.4),
    'd': (0.4, 0.6),
}


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


def place_station_below(satellite_position, epoch, rng, max_offset_angle=np.pi / 4):
    '''
    Place an Earth-fixed observing station on the ground near the point the
    satellite is passing over.

    The placement geometry is data_generation.py's: start from the satellite's
    radial direction, tilt it by a random angle of at most max_offset_angle
    about a random axis, and call that the station's direction. Because the
    station lands within 45 degrees of the sub-satellite point, it can see the
    object by construction -- no orbit ever goes unobserved.

    What differs from data_generation.py is what happens to that direction. It
    only means anything in GCRS at this one epoch, so a site built straight
    from it is frozen in inertial space and has no velocity at all. Converting
    it to ITRS -- the frame that co-rotates with Earth -- reinterprets it as a
    fixed latitude/longitude on the ground instead. That single reinterpretation
    is what supplies the velocity: a point fixed to a rotating rigid body moves
    at omega_earth x r, so get_gcrs_posvel can hand back both position and a
    consistent velocity at any later epoch.

    Parameters
    ----------
    satellite_position : np.array (3,)
        Satellite GCRS position at epoch (kilometers).
    epoch : astropy.time.Time
        Time at which the station is placed beneath the satellite.
    rng : np.random.Generator
        Source of the random tilt.
    max_offset_angle : float
        Largest tilt away from the satellite's direction (radians).

    Returns
    -------
    station : astropy.coordinates.EarthLocation
        Earth-fixed observing station.
    '''
    # Radial direction of the satellite, which is where the old file started.
    direction = satellite_position / np.linalg.norm(satellite_position)

    # Tilt it off the sub-satellite point. from_rotvec takes a rotation vector:
    # direction = rotation axis, magnitude = rotation angle.
    rotation_axis = rng.uniform(-1, 1, size=3)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_angle = rng.uniform(0, max_offset_angle)
    tilted_direction = R.from_rotvec(rotation_angle * rotation_axis).apply(direction)

    # Read that direction as a GCRS position at this epoch, then move it into
    # the Earth-fixed frame to find the latitude and longitude it points at.
    gcrs_point = GCRS(
        CartesianRepresentation(*(tilted_direction * R_EARTH_KM), unit=u.km),
        obstime=epoch
    )
    itrs_point = gcrs_point.transform_to(ITRS(obstime=epoch))

    # EarthLocation.geodetic returns (lon, lat, height) in that order. The
    # height is dropped and the station rebuilt at ground level: the direction
    # above was scaled by Earth's *equatorial* radius, which sits roughly 11 km
    # above the WGS84 ellipsoid at mid latitudes, and an 11 km-high telescope
    # is not worth simulating.
    longitude, latitude, _ = itrs_point.earth_location.geodetic
    station = EarthLocation.from_geodetic(lon=longitude, lat=latitude, height=0.0 * u.m)

    return station



if __name__ == "__main__":
    # Keep all four orbit choices; use near-GEO option A for now.
    test_case = 'a'
    this_dir = Path(__file__).resolve().parent

    clean_output_file = this_dir / 'generated_data_clean.txt'
    clean_npz_file = this_dir / "generated_data_clean.npz"

    noisy_output_file = this_dir / 'generated_data_noisy.txt'
    noisy_npz_file = this_dir / "generated_data_noisy.npz"

    # Each run appends one new object rather than overwriting the files.
    # The clean and noisy datasets must be generated together so they
    # retain the same objects, stations, times and true states.
    if clean_npz_file.exists() != noisy_npz_file.exists():
        raise RuntimeError(
            "Clean and noisy NPZ files must either both exist "
            "or both not exist."
        )

    if clean_npz_file.exists():
        with np.load(clean_npz_file) as existing_clean:
            existing_clean_data = existing_clean["data"]
            existing_clean_true_angles = existing_clean["true_angles"]
            existing_case = str(existing_clean["test_case"]) if "test_case" in existing_clean else test_case

        with np.load(noisy_npz_file) as existing_noisy:
            existing_noisy_data = existing_noisy["data"]
            existing_noisy_true_angles = existing_noisy["true_angles"]

        # Mixing populations in one dataset would make the admissible
        # region wrong for some objects and right for others, with no way
        # for a later stage to tell which.
        if existing_case != test_case:
            raise RuntimeError(
                f"{clean_npz_file.name} was generated with test_case "
                f"'{existing_case}', but this run uses '{test_case}'. "
                "Delete the generated_data_* files to start a new dataset."
            )

        object_id = int(existing_clean_data[:, 16].max()) + 1
    else:
        existing_clean_data = None
        existing_clean_true_angles = None

        existing_noisy_data = None
        existing_noisy_true_angles = None

        object_id = 0

    rng = np.random.default_rng(object_id)

    if test_case not in ECCENTRICITY_LIMITS:
        raise ValueError(
            "test_case must be 'a', 'b', 'c', or 'd'"
        )

    semi_major_axis_min, semi_major_axis_max = SEMI_MAJOR_AXIS_LIMITS_KM
    eccentricity_min, eccentricity_max = ECCENTRICITY_LIMITS[test_case]

    semi_major_axis = rng.uniform(
        semi_major_axis_min,
        semi_major_axis_max,
    )

    # Choose the semi-major axis first, then convert it to perigee radius,
    # which is what the classical-element vector below expects.
    eccentricity = rng.uniform(
        eccentricity_min,
        eccentricity_max,
    )

    perigee_radius = semi_major_axis * (1 - eccentricity)

    coe_oe = np.array([
        perigee_radius,
        eccentricity,
        rng.uniform(np.deg2rad(1), np.deg2rad(179)),
        rng.uniform(0, 2 * np.pi),
        rng.uniform(0, 2 * np.pi),
        rng.uniform(0, 2 * np.pi)
    ])

    # Siminski et al.: tracklets last 1-2 minutes and contain
    # 3-5 measurements.
    num_of_tracklets = 3
    points_per_tracklet = 5
    measurement_interval = 30.0

    tracklet_duration = (
        points_per_tracklet - 1
    ) * measurement_interval

    # The paper studies multiple separation-time groups rather than
    # requiring one exact gap. Near-GEO periods are ~20.5-26.4 h, so this
    # gap sets how many revolutions k separate consecutive tracklets:
    # 12 h -> k in {0, 0.5}, 24 h -> {0.5, 1.0}, 36 h -> {1.0, 1.5},
    # 48 h -> {1.5, 2.0}. hypothesis/k_bounds.py derives the actual grid
    # from the resulting epochs, so only this line changes.
    gap_between_tracklets = 36.0 * 60.0 * 60.0

    tracklet_start_interval = (
        tracklet_duration + gap_between_tracklets
    )

    # Siminski et al. measurement-error cases: relative angular noise of
    # 0.7, 1.2, or 2.0 arcsec and a 5 arcsec systematic-error test.
    relative_uncertainty_arcsec = 0.7
    systematic_uncertainty_arcsec = 5.0

    arcsec_to_rad = np.deg2rad(1.0 / 3600.0)

    sigma_relative_rad = (
        relative_uncertainty_arcsec * arcsec_to_rad
    )

    sigma_systematic_rad = (
        systematic_uncertainty_arcsec * arcsec_to_rad
    )

    start_epoch = Time(
        '2026-07-21T00:00:00',
        scale='utc'
    )

    eq_oe = coe_to_eq_oe(coe_oe)
    rv_init = oe_to_gcrs(eq_oe)[np.newaxis, :]

    # Observing network: one station per tracklet, each placed on the
    # ground beneath the object at that tracklet's own reference time,
    # and used only for that tracklet.
    tracklet_reference_times = [
        tracklet_id * tracklet_start_interval
        for tracklet_id in range(num_of_tracklets)
    ]

    station_locations = []

    for reference_time in tracklet_reference_times:
        r_reference, _ = two_body(
            mu,
            reference_time,
            rv_init[0, 0:3],
            rv_init[0, 3:6]
        )

        station_locations.append(
            place_station_below(
                r_reference,
                start_epoch + reference_time * u.s,
                rng
            )
        )

    # Generate the clean and noisy versions from the same orbit,
    # stations and observation times.
    clean_rows = []
    noisy_rows = []
    true_angle_rows = []

    for tracklet_id in range(num_of_tracklets):
        # One constant RA bias and one constant Dec bias for this tracklet.
        # These are used only in the noisy dataset.
        ra_systematic_error = rng.normal(
            0.0,
            sigma_systematic_rad
        )

        dec_systematic_error = rng.normal(
            0.0,
            sigma_systematic_rad
        )

        # Each tracklet uses its own station.
        site_location = station_locations[tracklet_id]

        for point_id in range(points_per_tracklet):
            measurement_time = (
                tracklet_id * tracklet_start_interval
                + point_id * measurement_interval
            )

            r, v = two_body(
                mu,
                measurement_time,
                rv_init[0, 0:3],
                rv_init[0, 3:6]
            )

            epoch = (
                start_epoch
                + measurement_time * u.s
            )

            site_position, site_velocity = (
                site_location.get_gcrs_posvel(epoch)
            )

            site_position_km = (
                site_position.xyz.to_value(u.km)
            )

            site_velocity_km_s = (
                site_velocity.xyz.to_value(u.km / u.s)
            )

            line_of_sight = r - site_position_km

            ra_true, dec_true = gcrs_to_radec(
                line_of_sight
            )

            # Independent relative noise is redrawn for every
            # RA/Dec observation.
            ra = (
                ra_true
                + ra_systematic_error
                + rng.normal(0.0, sigma_relative_rad)
            ) % (2.0 * np.pi)

            dec = (
                dec_true
                + dec_systematic_error
                + rng.normal(0.0, sigma_relative_rad)
            )

            true_angle_rows.append([
                ra_true,
                dec_true,
                measurement_time,
                tracklet_id,
                object_id
            ])

            # Perfect-observation row.
            clean_rows.append([
                *r,                         # columns 0-2
                *v,                         # columns 3-5
                ra_true,                    # column 6: noise-free RA
                dec_true,                   # column 7: noise-free Dec
                *site_position_km,          # columns 8-10
                *site_velocity_km_s,        # columns 11-13
                measurement_time,           # column 14
                tracklet_id,                # column 15
                object_id                   # column 16
            ])

            # Noisy-observation row.
            noisy_rows.append([
                *r,                         # columns 0-2
                *v,                         # columns 3-5
                ra,                         # column 6: noisy RA
                dec,                        # column 7: noisy Dec
                *site_position_km,          # columns 8-10
                *site_velocity_km_s,        # columns 11-13
                measurement_time,           # column 14
                tracklet_id,                # column 15
                object_id                   # column 16
            ])

    new_clean_data = np.asarray(
        clean_rows,
        dtype=np.float64
    )

    new_noisy_data = np.asarray(
        noisy_rows,
        dtype=np.float64
    )

    new_true_angles = np.asarray(
        true_angle_rows,
        dtype=np.float64
    )

    # Append the new object to the existing clean and noisy datasets.
    if existing_clean_data is not None:
        generated_clean_data = np.vstack([
            existing_clean_data,
            new_clean_data
        ])

        generated_noisy_data = np.vstack([
            existing_noisy_data,
            new_noisy_data
        ])

        clean_true_angles = np.vstack([
            existing_clean_true_angles,
            new_true_angles
        ])

        noisy_true_angles = np.vstack([
            existing_noisy_true_angles,
            new_true_angles
        ])

    else:
        generated_clean_data = new_clean_data
        generated_noisy_data = new_noisy_data

        clean_true_angles = new_true_angles
        noisy_true_angles = new_true_angles.copy()

    # Canonical calculation file for perfectly observed data.
    np.savez(
        clean_npz_file,
        data=generated_clean_data,
        true_angles=clean_true_angles,
        relative_uncertainty_arcsec=np.float64(0.0),
        systematic_uncertainty_arcsec=np.float64(0.0),
        random_seed=np.int64(object_id),
        test_case=test_case,
        a_min_km=np.float64(semi_major_axis_min),
        a_max_km=np.float64(semi_major_axis_max),
        e_min=np.float64(eccentricity_min),
        e_max=np.float64(eccentricity_max)
    )

    # Canonical calculation file for noisy observed data.
    np.savez(
        noisy_npz_file,
        data=generated_noisy_data,
        true_angles=noisy_true_angles,
        relative_uncertainty_arcsec=np.float64(
            relative_uncertainty_arcsec
        ),
        systematic_uncertainty_arcsec=np.float64(
            systematic_uncertainty_arcsec
        ),
        random_seed=np.int64(object_id),
        test_case=test_case,
        a_min_km=np.float64(semi_major_axis_min),
        a_max_km=np.float64(semi_major_axis_max),
        e_min=np.float64(eccentricity_min),
        e_max=np.float64(eccentricity_max)
    )

    # Readable version of the clean dataset.
    np.savetxt(
        clean_output_file,
        generated_clean_data,
        delimiter=",",
        fmt=[
            "%f", "%f", "%f",        # satellite position (km)
            "%f", "%f", "%f",        # satellite velocity (km/s)
            "%f", "%f",              # RA and Dec (rad)
            "%f", "%f", "%f",        # station position (km)
            "%f", "%f", "%f",        # station velocity (km/s)
            "%f",                     # measurement time (s)
            "%d", "%d"                # tracklet and object IDs
        ],
        header=(
            "truth_rx_km,truth_ry_km,truth_rz_km,"
            "truth_vx_km_s,truth_vy_km_s,truth_vz_km_s,"
            "ra_rad,dec_rad,"
            "site_x_km,site_y_km,site_z_km,"
            "site_vx_km_s,site_vy_km_s,site_vz_km_s,"
            "time_s,tracklet_id,object_id"
        )
    )

    # Readable version of the noisy dataset.
    np.savetxt(
        noisy_output_file,
        generated_noisy_data,
        delimiter=",",
        fmt=[
            "%f", "%f", "%f",        # satellite position (km)
            "%f", "%f", "%f",        # satellite velocity (km/s)
            "%f", "%f",              # RA and Dec (rad)
            "%f", "%f", "%f",        # station position (km)
            "%f", "%f", "%f",        # station velocity (km/s)
            "%f",                     # measurement time (s)
            "%d", "%d"                # tracklet and object IDs
        ],
        header=(
            "truth_rx_km,truth_ry_km,truth_rz_km,"
            "truth_vx_km_s,truth_vy_km_s,truth_vz_km_s,"
            "ra_rad,dec_rad,"
            "site_x_km,site_y_km,site_z_km,"
            "site_vx_km_s,site_vy_km_s,site_vz_km_s,"
            "time_s,tracklet_id,object_id"
        )
    )