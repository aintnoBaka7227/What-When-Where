import numpy as np
from numba import jit
from scipy.spatial.transform import Rotation as R
from astropy.constants import GM_earth, R_earth


GM_EARTH = GM_earth.value
R_EARTH = R_earth.value
mu = GM_EARTH / 1000000000


def gcrs_to_coe(gcrs):
    r = gcrs[0:3]
    v = gcrs[3:6]
    rmag = np.linalg.norm(r)
    vmag = np.linalg.norm(v)

    epsilon = (vmag * vmag / 2) - (mu / rmag)
    a = 1 / (-2 * epsilon / mu)

    rhat = r / rmag
    hvec = np.cross(r, v)
    hmag = np.linalg.norm(hvec)
    hhat = hvec / hmag

    ecc = np.cross(v, hvec) / mu - rhat
    e = np.linalg.norm(ecc)

    khat = np.array([0, 0, 1])
    i = np.arccos(np.dot(khat, hhat))

    ihat = np.array([1, 0, 0])
    nvec = np.cross(khat, hvec)
    nhat = nvec / np.linalg.norm(nvec)
    if (nhat[1] < 0):
        ohm = 2 * np.pi - np.arccos(np.dot(ihat, nhat))
    else:
        ohm = np.arccos(np.dot(ihat, nhat))

    if (ecc[2] < 0):
        omega = 2 * np.pi - np.arccos(np.dot(nhat, ecc / e))
    else:
        omega = np.arccos(np.dot(nhat, ecc / e))

    if (np.dot(r, v) < 0):
        v = 2 * np.pi - np.arccos(np.dot(ecc / e, rhat))
    else:
        v = np.arccos(np.dot(ecc / e, rhat))

    rp = a * (1 - e)
    coe = np.array([rp, e, i, ohm, omega, v])

    return coe


def coe_to_gcrs(coe):
    rp = coe[0]  # takes km
    e = coe[1]
    i = coe[2]
    ohm = coe[3]
    omega = coe[4]
    v = coe[5]

    a = rp / (1 - e)
    p = a * (1 - e * e)

    r_pqw_x = (p * np.cos(v)) / (1 + e * np.cos(v))
    r_pqw_y = (p * np.sin(v)) / (1 + e * np.cos(v))
    r_pqw = np.array([r_pqw_x, r_pqw_y, 0])

    v_pqw_x = - np.sqrt(mu / p) * np.sin(v)
    v_pqw_y = np.sqrt(mu / p) * (e + np.cos(v))

    v_pqw = np.array([v_pqw_x, v_pqw_y, 0])

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
    # Assign the elements
    a, ex, ey, hx, hy, l = oe.T.copy()

    # Shorthand terms
    p = a * (1 - (ex ** 2 + ey ** 2))
    s2 = 1 + hx ** 2 + hy ** 2
    alpha2 = hx ** 2 - hy ** 2
    r = p / (1 + ex * np.cos(l) + ey * np.sin(l))

    # Positions
    px = np.cos(l) + alpha2 * np.cos(l) + 2 * hx * hy * np.sin(l)
    py = np.sin(l) - alpha2 * np.sin(l) + 2 * hx * hy * np.cos(l)
    pz = 2 * (hx * np.sin(l) - hy * np.cos(l))
    pos = r / s2 * np.array([px, py, pz])

    # Velocities
    vx = np.sin(l) + alpha2 * np.sin(l) - 2 * hx * hy * np.cos(l) + ey - 2 * ex * hx * hy + alpha2 * ey
    vy = -np.cos(l) + alpha2 * np.cos(l) + 2 * hx * hy * np.sin(l) - ex + 2 * ey * hx * hy + alpha2 * ex
    vz = -2 * (hx * np.cos(l) + hy * np.sin(l) + ex * hx + ey * hy)
    # vel = -1 / s2 * np.sqrt(GM_EARTH / p) * np.array([vx, vy, vz])
    vel = -1 / s2 * np.sqrt(mu / p) * np.array([vx, vy, vz])

    # Construct the position/velocity state matrix
    gcrs = np.array([pos[0], pos[1], pos[2],
                     vel[0], vel[1], vel[2]]).T

    return gcrs


def gcrs_to_oe(gcrs):
    r = gcrs[0:3]
    v = gcrs[3:6]
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

    X1 = np.dot(r, fhat)
    Y1 = np.dot(r, ghat)

    epsilon = (vmag * vmag / 2) - (mu / rmag)
    a = 1 / (-2 * epsilon / mu)

    L = np.arctan2(rhat[1] - vhat[0], rhat[0] + vhat[1])
    oe = np.array([a, f, g, h, k, L])

    return oe


def coe_to_eq_oe(coe):
    rp = coe[0]  # takes km
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
    lv = v + omega + ohm

    eq_oe = np.array([a, ex, ey, hx, hy, lv])
    return eq_oe


@jit(nopython=True, nogil=True, cache=True)
def two_body(mu, tau, ri, vi):
    """
    :param mu: gravitational constant(km ** 3 / sec ** 2)
    :param tau: propagation time interval(seconds)
    :param ri: initial eci position vector(kilometers)
    :param vi: initial eci velocity vector(kilometers / second)
    :return:
    rf = final eci position vector(kilometers)
    vf = final eci velocity vector(kilometers / second)
    """
    tolerance = 1.0e-12
    u = np.float64(0.0)
    uold = 100
    dtold = 100
    # imax = 20
    imax = 100

    # umax = sys.float_info.max
    # umin = -sys.float_info.max
    umax = np.float64(1.7976931348623157e+308)
    umin = np.float64(-umax)

    orbits = 0

    tdesired = tau

    threshold = tolerance * abs(tdesired)

    r0 = np.linalg.norm(ri)

    n0 = np.dot(ri, vi)

    beta = 2 * (mu / r0) - np.dot(vi, vi)

    if (beta != 0):
        umax = +1 / np.sqrt(abs(beta))
        umin = -1 / np.sqrt(abs(beta))

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

        if (i == imax):
            print('max iterations in twobody2 function')

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
    # Separate and assign the gcrs components
    px, py, pz = gcrs.T[:3]

    # Convert positions to gcrs
    rho = np.sqrt(px ** 2 + py ** 2 + pz ** 2)

    ra = np.arctan2(py, px)
    if ra < 0:
        ra = 2 * np.pi + ra

    dec = np.arcsin(pz / rho)

    radec = np.array([ra, dec]).T

    return radec




def radec_to_point(radec):
    point = np.array([np.cos(radec[1]) * np.cos((radec[0])),
                      np.cos(radec[1]) * np.sin(radec[0]),
                      np.sin(radec[1])])

    return point



if __name__ == "__main__":
    
    test_case = 'a'
    output_dir = 'generated_data.txt'

    
    if test_case == 'a': #different orbital test cases
        coe_oe = np.array([np.random.uniform(6880, 8380),
                        # 1e-8, 1e-8,  # Circular orbit
                        np.random.uniform(0, 0.01), #e
                        # 0,
                        np.random.uniform(np.deg2rad(1), np.deg2rad(179)), #i
                        np.random.uniform(0, 2 * np.pi), #ohm
                        np.random.uniform(0, 2 * np.pi), #omega
                        np.random.uniform(0, 2 * np.pi)]) #v
    elif test_case == 'b':
        coe_oe = np.array([np.random.uniform(8380, 9380),
                            # 1e-8, 1e-8,  # Circular orbit
                            np.random.uniform(0.01, 0.2),  # e
                            # 0,
                            np.random.uniform(np.deg2rad(1), np.deg2rad(179)),  # i
                            np.random.uniform(0, 2 * np.pi),  # ohm
                            np.random.uniform(0, 2 * np.pi),  # omega
                            np.random.uniform(0, 2 * np.pi)])  # v
    elif test_case == 'c':
        coe_oe = np.array([np.random.uniform(8380, 9380),
                            # 1e-8, 1e-8,  # Circular orbit
                            np.random.uniform(0.2, 0.4),  # e
                            # 0,
                            np.random.uniform(np.deg2rad(1), np.deg2rad(179)),  # i
                            np.random.uniform(0, 2 * np.pi),  # ohm
                            np.random.uniform(0, 2 * np.pi),  # omega
                            np.random.uniform(0, 2 * np.pi)])  # v
    elif test_case == 'd':
        coe_oe = np.array([np.random.uniform(8380, 9380),
                            # 1e-8, 1e-8,  # Circular orbit
                            np.random.uniform(0.4, 0.6),  # e
                            # 0,
                            np.random.uniform(np.deg2rad(1), np.deg2rad(179)),  # i
                            np.random.uniform(0, 2 * np.pi),  # ohm
                            np.random.uniform(0, 2 * np.pi),  # omega
                            np.random.uniform(0, 2 * np.pi)])  # v


    # let the user set the number of locations
    num_of_sites = 3
    time_interval_between_sites = 120 #seconds
    num_of_pts_in_each_site = 5
    measurement_time_in_each_site = 5 #seconds
    measurement_time_interval_within_each_site = measurement_time_in_each_site / num_of_pts_in_each_site

    # propagate the orbit to the num_of_loc locations
    # convert orbital state to position and velocity vectors
    eq_oe = coe_to_eq_oe(coe_oe)
    rv_init = oe_to_gcrs(eq_oe)
    rv_init = rv_init[np.newaxis, :]

    for i in range(num_of_sites):
        # propagate
        for j in range(num_of_pts_in_each_site):    
            r, v = two_body(mu, i * time_interval_between_sites + j * measurement_time_interval_within_each_site, rv_init[0, 0:3], rv_init[0, 3:6])
            rv = np.zeros_like(rv_init)
            rv[0, 0:3] = r
            rv[0, 3:6] = v
            
            if j == 0: # setup the rsite
                rsite = rv[0, 0:3] / np.linalg.norm(rv[0, 0:3])
                # A way to generate the observing site from the radial direction. 
                rot_axis = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
                rot_axis = rot_axis / np.linalg.norm(rot_axis)
                rot_angle = np.random.uniform(0, (np.pi / 4))
                curr_R = R.from_rotvec(rot_angle * rot_axis)
                curr_R_mat = curr_R.as_matrix()

                rotated_rsite = np.matmul(curr_R_mat, rsite)
                rotated_rsite = rotated_rsite * (R_EARTH)
            
            radec = gcrs_to_radec(rv[0, 0:3] - rotated_rsite / 1000)

            # write rv, radec, rotated_rsite, and time into the text file
            with open(output_dir, 'a') as f:
                f.write(f"{rv[0,0]},{rv[0,1]},{rv[0,2]},{rv[0,3]},{rv[0,4]},{rv[0,5]},{radec[0]},{radec[1]},{rotated_rsite[0]},{rotated_rsite[1]},{rotated_rsite[2]},{i * time_interval_between_sites + j * measurement_time_interval_within_each_site}\n")
                
