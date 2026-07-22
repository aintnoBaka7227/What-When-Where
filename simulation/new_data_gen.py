import numpy as np
from numba import jit
from scipy.spatial.transform import Rotation as R
from astropy.constants import GM_earth, R_earth
import astropy.units as u
from astropy.coordinates import GCRS, ITRS, EarthLocation, CartesianRepresentation
from astropy.time import Time
from astropy.utils import iers


iers.conf.auto_download = False


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



if __name__ == "__main__":
    # Keep all four orbit choices; use near-GEO option A for now.
    test_case = 'a'
    output_dir = 'generated_data.txt'

    if test_case == 'a':
        # Near-GEO: choose semi-major axis first, then convert it to perigee.
        semi_major_axis = np.random.uniform(38000, 45000)
        eccentricity = np.random.uniform(0, 0.01)
        perigee_radius = semi_major_axis * (1 - eccentricity)
        coe_oe = np.array([
            perigee_radius,
            eccentricity,
            np.random.uniform(np.deg2rad(1), np.deg2rad(179)),
            np.random.uniform(0, 2 * np.pi),
            np.random.uniform(0, 2 * np.pi),
            np.random.uniform(0, 2 * np.pi)
        ])
    elif test_case == 'b':
        coe_oe = np.array([np.random.uniform(8380, 9380),
                           np.random.uniform(0.01, 0.2),
                           np.random.uniform(np.deg2rad(1), np.deg2rad(179)),
                           np.random.uniform(0, 2 * np.pi),
                           np.random.uniform(0, 2 * np.pi),
                           np.random.uniform(0, 2 * np.pi)])
    elif test_case == 'c':
        coe_oe = np.array([np.random.uniform(8380, 9380),
                           np.random.uniform(0.2, 0.4),
                           np.random.uniform(np.deg2rad(1), np.deg2rad(179)),
                           np.random.uniform(0, 2 * np.pi),
                           np.random.uniform(0, 2 * np.pi),
                           np.random.uniform(0, 2 * np.pi)])
    elif test_case == 'd':
        coe_oe = np.array([np.random.uniform(8380, 9380),
                           np.random.uniform(0.4, 0.6),
                           np.random.uniform(np.deg2rad(1), np.deg2rad(179)),
                           np.random.uniform(0, 2 * np.pi),
                           np.random.uniform(0, 2 * np.pi),
                           np.random.uniform(0, 2 * np.pi)])
    else:
        raise ValueError("test_case must be 'a', 'b', 'c', or 'd'")

    num_of_tracklets = 3
    time_interval_between_tracklets = 120
    points_per_tracklet = 5
    measurement_interval = 5 / points_per_tracklet
    start_epoch = Time('2026-07-21T00:00:00', scale='utc')
    object_id = 0

    eq_oe = coe_to_eq_oe(coe_oe)
    rv_init = oe_to_gcrs(eq_oe)[np.newaxis, :]

    # Choose one convenient station near the object's initial direction.
    site_direction = rv_init[0, 0:3] / np.linalg.norm(rv_init[0, 0:3])
    rot_axis = np.random.uniform(-1.0, 1.0, 3)
    rot_axis /= np.linalg.norm(rot_axis)
    site_gcrs_m = R.from_rotvec(
        np.random.uniform(0.0, np.deg2rad(30.0)) * rot_axis
    ).apply(site_direction) * R_EARTH

    # Fix that station to Earth and reuse it for every tracklet.
    site_gcrs = GCRS(
        CartesianRepresentation(*(site_gcrs_m * u.m)),
        obstime=start_epoch
    )
    site_itrs = site_gcrs.transform_to(ITRS(obstime=start_epoch))
    site_location = EarthLocation.from_geocentric(
        site_itrs.x, site_itrs.y, site_itrs.z
    )

    raw_rows = []

    for tracklet_id in range(num_of_tracklets):
        for point_id in range(points_per_tracklet):
            measurement_time = (
                tracklet_id * time_interval_between_tracklets
                + point_id * measurement_interval
            )

            r, v = two_body(
                mu, measurement_time,
                rv_init[0, 0:3], rv_init[0, 3:6]
            )

            epoch = start_epoch + measurement_time * u.s
            site_position, site_velocity = site_location.get_gcrs_posvel(epoch)
            site_position_km = site_position.xyz.to_value(u.km)
            site_velocity_km_s = site_velocity.xyz.to_value(u.km / u.s)

            ra, dec = gcrs_to_radec(r - site_position_km)
            raw_rows.append([
                *r, *v, ra, dec,
                *site_position_km, *site_velocity_km_s,
                measurement_time, tracklet_id, object_id
            ])

    # Raw tracklet samples only; fit attributables in the separate fitting file.
    np.savetxt(output_dir, np.asarray(raw_rows), delimiter=',')
                
