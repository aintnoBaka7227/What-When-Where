import numpy as np


def fit_attributes(
    times,
    ra,
    dec,
    relative_uncertainty_arcsec,
    systematic_uncertainty_arcsec
):
    """
    Compress one tracklet into an attributable, as in Siminski et al.

    Fits angle(t) = angle(t0) + rate * (t - t0) to RA and Dec separately, by
    least squares.

    Parameters
    ----------
    times : array-like
        Observation times, s.
    ra, dec : array-like
        Measured angles, rad.
    relative_uncertainty_arcsec : float
        Point-to-point noise (the paper's 0.7", 1.2", 2.0"). Sets the
        covariance of the fit, including the rates.
    systematic_uncertainty_arcsec : float
        A constant whole-tracklet offset. Added to the angle variances only,
        since a constant shift does not change a fitted slope.

    Returns
    -------
    a : ndarray, shape (4,)
        [alpha, alpha_dot, delta, delta_dot], angles at t0.
    Ca : ndarray, shape (4, 4)
        Covariance of a, same ordering.
    t0 : float
        Reference epoch, the mean observation time, s.
    """

    times = np.asarray(times, dtype=float)
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)

    # Remove 2*pi jumps. alpha may then lie outside [0, 2*pi).
    ra = np.unwrap(ra)

    # Centring on the mean time decorrelates the fitted angle and rate.
    t0 = np.mean(times)
    dt = times - t0

    # Columns: intercept (angle at t0), slope (rate).
    X = np.column_stack((np.ones_like(dt), dt))

    beta_ra = np.linalg.lstsq(X, ra, rcond=None)[0]
    beta_dec = np.linalg.lstsq(X, dec, rcond=None)[0]

    sigma_relative = np.deg2rad(
        relative_uncertainty_arcsec / 3600.0
    )

    sigma_systematic = np.deg2rad(
        systematic_uncertainty_arcsec / 3600.0
    )

    # C_beta = sigma^2 (X^T X)^-1. Units: rad^2, rad^2/s, rad^2/s^2.
    fit_geometry_covariance = np.linalg.inv(X.T @ X)

    cov_ra = sigma_relative**2 * fit_geometry_covariance
    cov_dec = sigma_relative**2 * fit_geometry_covariance

    # The systematic offset moves the angles, not the rates.
    cov_ra[0, 0] += sigma_systematic**2
    cov_dec[0, 0] += sigma_systematic**2

    a = np.array([
        beta_ra[0],
        beta_ra[1],
        beta_dec[0],
        beta_dec[1]
    ])

    # Block diagonal: RA and Dec errors are independent.
    Ca = np.zeros((4, 4))

    Ca[0:2, 0:2] = cov_ra
    Ca[2:4, 2:4] = cov_dec

    return a, Ca, t0
