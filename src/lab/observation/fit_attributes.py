import numpy as np


# A tracklet is a short sequence of angular observations of one object.
# This function replaces that sequence with an attributable: the fitted
# direction and angular rate at one reference epoch.
def fit_attributes(
    times,
    ra,
    dec,
    relative_uncertainty_arcsec,
    systematic_uncertainty_arcsec
):
    """
    Fit the attributable vector used by Siminski et al.

    The attributable follows the exact ordering used in the paper:

        a = [alpha, alpha_dot, delta, delta_dot]

    Parameters
    ----------
    times : array-like
        Observation times in seconds.

    ra : array-like
        Measured right ascension values in radians.

    dec : array-like
        Measured declination values in radians.

    relative_uncertainty_arcsec : float
        Point-to-point angular uncertainty within the tracklet, in arcseconds.

        This uncertainty controls the covariance of the least-squares fit,
        particularly the uncertainty in alpha_dot and delta_dot.

        The main-paper values tested include approximately:
            0.7 arcsec
            1.2 arcsec
            2.0 arcsec

    systematic_uncertainty_arcsec : float
        Whole-tracklet angular uncertainty in arcseconds.

        This represents an approximately constant angular offset affecting
        every observation in the tracklet. It is therefore added to the
        uncertainty of alpha and delta, but not directly to alpha_dot and
        delta_dot.

    Returns
    -------
    a : ndarray, shape (4,)
        Attributable vector:

            [alpha, alpha_dot, delta, delta_dot]

        alpha and delta are evaluated at the reference epoch t0.

    Ca : ndarray, shape (4, 4)
        Covariance matrix of the attributable vector, using the same ordering
        as a.

    t0 : float
        Reference epoch in seconds.
    """

    times = np.asarray(times, dtype=float)
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)

    # RA is a periodic angle. Unwrapping prevents a crossing from 2*pi to zero
    # from appearing as a large negative angular movement. The fitted alpha may
    # consequently lie outside [0, 2*pi), but it remains angularly equivalent.
    ra = np.unwrap(ra)

    # Use the mean observation time as the reference epoch.
    #
    # The fitted alpha and delta therefore describe the direction at the
    # middle of the tracklet. Centering time also reduces covariance between
    # the fitted angular positions and angular rates.
    t0 = np.mean(times)
    dt = times - t0

    # Design matrix for the linear angular-motion model:
    #
    #     angle(t) = angle(t0) + angle_rate * (t - t0)
    #
    # For example, with five observations:
    #
    #     X = [[1, dt_1],
    #          [1, dt_2],
    #          [1, dt_3],
    #          [1, dt_4],
    #          [1, dt_5]]
    #
    # The first column fits the angle at t0.
    # The second column fits the angular rate.
    X = np.column_stack((np.ones_like(dt), dt))

    # Fit right ascension:
    #
    #     beta_ra[0] = alpha
    #     beta_ra[1] = alpha_dot
    #
    # lstsq returns the intercept and slope that minimize the sum of squared
    # residuals between the observations and the linear model.
    beta_ra = np.linalg.lstsq(X, ra, rcond=None)[0]

    # Fit declination:
    #
    #     beta_dec[0] = delta
    #     beta_dec[1] = delta_dot
    beta_dec = np.linalg.lstsq(X, dec, rcond=None)[0]

    # Convert uncertainty values from arcseconds to radians.
    #
    # There are 3600 arcseconds in one degree.
    sigma_relative = np.deg2rad(
        relative_uncertainty_arcsec / 3600.0
    )

    sigma_systematic = np.deg2rad(
        systematic_uncertainty_arcsec / 3600.0
    )

    # For independent observations having the same relative uncertainty,
    # the covariance of the fitted intercept and slope is:
    #
    #     C_beta = sigma_relative^2 * (X^T X)^(-1)
    #
    # The diagonal entries represent variances:
    #
    #     C_beta[0, 0] = variance of fitted angle
    #     C_beta[1, 1] = variance of fitted angular rate
    #
    # The off-diagonal entries represent angle-rate covariance.
    #
    # Since dt is measured in seconds, the resulting entries have units:
    #   C[0,0] = rad^2
    #   C[0,1] = C[1,0] = rad^2/s
    #   C[1,1] = rad^2/s^2
    fit_geometry_covariance = np.linalg.inv(X.T @ X)

    cov_ra = sigma_relative**2 * fit_geometry_covariance
    cov_dec = sigma_relative**2 * fit_geometry_covariance

    # The systematic uncertainty shifts the whole tracklet.
    #
    # A constant shift changes the fitted angular position but does not change
    # the fitted slope. Its variance is consequently added only to the fitted
    # alpha and delta variances.
    cov_ra[0, 0] += sigma_systematic**2
    cov_dec[0, 0] += sigma_systematic**2

    # Construct the attributable using the exact paper ordering:
    #
    #     [alpha, alpha_dot, delta, delta_dot]
    a = np.array([
        beta_ra[0],
        beta_ra[1],
        beta_dec[0],
        beta_dec[1]
    ])

    # Construct the attributable covariance using the same ordering:
    #
    #             alpha  alpha_dot  delta  delta_dot
    #
    # alpha         x        x        0        0
    # alpha_dot     x        x        0        0
    # delta         0        0        x        x
    # delta_dot     0        0        x        x
    #
    # This assumes that the RA and Dec measurement errors are independent.
    Ca = np.zeros((4, 4))

    Ca[0:2, 0:2] = cov_ra
    Ca[2:4, 2:4] = cov_dec

    # Return the fitted attributable, its uncertainty, and the epoch at which
    # the fitted angular positions alpha and delta are defined.
    return a, Ca, t0
