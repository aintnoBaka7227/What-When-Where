"""Minimize the discriminator loss over a range hypothesis (Eq. 30-31).

For a fixed k, find

    p_k_min = argmin_{p in C} L(p, k)                                  Eq. (30)

where p = (rho_1, rho_2), C is the admissible region (Eq. 22, i.e. each
rho_i confined to src/lab/hypothesis/range_bounds.py's [rho_min_i, rho_max_i]), and
L is src/lab/scoring/discriminator_loss.py's loss. The search is seeded from
the circular-orbit initial guess

    rho_i* = -c_i + sqrt(c_i^2 + a(k)^2 - ||r_s_i||^2)                 Eq. (31)

using a semi-major axis a(k) from Kepler's third law. As printed a(k) is
ill-defined at k = 0, and each k spans a *range* of swept revolutions,
[k, k+0.5) per Eqs. (29)-(30). This uses the bin midpoint k + 0.25, with
k = 0 falling back to the population midpoint -- see
semi_major_axis_for_k().

discriminator_loss() returns one loss per Lambert candidate (the vacant-
focus ambiguity for M > 0, see src/lab/model/lambert.py). The optimizer reduces
that to a scalar via min() at every trial point, letting the search pick
whichever candidate fits best rather than committing to one branch label
across possibly-different trial points.

The paper specifies quasi-Newton search with numerically approximated
derivatives when no analytic gradient is known (Section 5.2) -- exactly
what scipy.optimize.minimize(method="L-BFGS-B") does without an explicit
`jac`, so no analytic gradient of L is derived here. Its finite-difference
step and tolerances are set explicitly (see _MINIMIZER_OPTIONS); the
defaults under-resolve the loss valley and stop the search early.
"""


import numpy as np
from astropy.constants import GM_earth
from scipy.optimize import minimize

from lab.data.dataset import VARIANTS, data_file, load_data
from lab.hypothesis.range_bounds import (
    ASSOCIATION_FILES,
    ATTRIBUTABLES_FILES,
    A_MIN,
    A_MAX,
    population_for,
)
from lab.hypothesis.k_bounds import k_grid
from lab.model.build_orbit import (
    RANGE_BOUNDS_FILES,
    K_BOUNDS_FILES,
    _lookup_tracklet,
)
from lab.model.lambert import LambertNoSolution
from lab.scoring.discriminator_loss import LOSS_FUNCTIONS, evaluate_loss

# km^3/s^2, matches the rest of the project.
MU = GM_earth.value / 1_000_000_000.0

# A trial hypothesis that fails to produce any Lambert solution is heavily
# penalized (rather than raising out of the optimizer) so L-BFGS-B can step
# away from it instead of crashing the search.
_INFEASIBLE_PENALTY = 1.0e12

# L-BFGS-B options for the Eq. (30) search over p = (rho_1, rho_2).
#
# The defaults fail here. `eps` -- the finite-difference step for the
# numerical gradient -- defaults to ~1e-8, so on ranges of ~4e4 km the
# gradient is probed across a step 12 orders of magnitude too small and
# comes back as rounding noise. L-BFGS-B then "converges" two iterations
# from the circular-orbit seed, thousands of times above the true minimum
# (src/lab/evaluation/validate_optimiser.py demonstrates this). A 1 km step
# resolves the loss valley, which stays ~500 km wide even for the
# ill-conditioned near-180-degree transfers. ftol/gtol are tightened to
# match and maxiter is raised so the tighter tolerances have room to run.
#
# This `eps` is unrelated to the `eps` argument threaded through to
# discriminator_loss for the Eq. (19) Jacobian -- different finite
# difference, over a different variable.
_MINIMIZER_OPTIONS = {
    "eps": 1.0,
    "ftol": 1.0e-11,
    "gtol": 1.0e-10,
    "maxiter": 500,
}


def semi_major_axis_for_k(
    k,
    delta_t,
    mu=MU,
    a_min=A_MIN,
    a_max=A_MAX,
):
    """Calculate an admissible circular-orbit seed a(k)."""
    delta_t = abs(delta_t)

    if delta_t == 0.0:
        raise ValueError("Tracklet epochs must be different")

    # Calculate periods at the allowed semi-major-axis boundaries.
    period_a_min = 2.0 * np.pi * np.sqrt(a_min**3 / mu)
    period_a_max = 2.0 * np.pi * np.sqrt(a_max**3 / mu)

    # k labels [k, k + 0.5), not one exact revolution count.
    # Use the midpoint of the part overlapping [a_min, a_max].
    revolutions_min = delta_t / period_a_max
    revolutions_max = delta_t / period_a_min

    lower = max(k, revolutions_min)
    upper = min(k + 0.5, revolutions_max)

    if lower >= upper:
        raise ValueError(
            f"k={k} does not overlap [{a_min}, {a_max}] km"
        )

    eps = 1e-3 * (upper - lower)

    revolutions_star = np.clip(k,lower + eps,upper - eps,)
    period = delta_t / revolutions_star

    a_k = (
        mu * period**2 / (4.0 * np.pi**2)
    ) ** (1.0 / 3.0)

    if not (a_min <= a_k <= a_max):
        raise ValueError(
            f"a({k})={a_k:.3f} km is outside "
            f"[{a_min}, {a_max}] km"
        )

    return a_k


def circular_orbit_initial_guess(
    k, delta_t, c_1, r_s1, c_2, r_s2, mu=MU, a_min=A_MIN, a_max=A_MAX,
):
    """Equation (31): rho_i* assuming a circular orbit of semi-major axis a(k)."""
    a_k = semi_major_axis_for_k(k, delta_t, mu, a_min, a_max)

    r_s1_norm_sq = np.dot(r_s1, r_s1)
    r_s2_norm_sq = np.dot(r_s2, r_s2)

    # Clipped at 0 defensively: for a(k) implausibly close to the site
    # distance the radicand can go slightly negative from floating point
    # error; the resulting rho* is only ever used as a search seed.
    rho_1_star = -c_1 + np.sqrt(max(c_1**2 + a_k**2 - r_s1_norm_sq, 0.0))
    rho_2_star = -c_2 + np.sqrt(max(c_2**2 + a_k**2 - r_s2_norm_sq, 0.0))

    return rho_1_star, rho_2_star


def _objective(p, variant, k, r_s1, v_s1, t1, attributable_1, ca_1, r_s2, v_s2, t2, attributable_2, ca_2, mu, eps):
    rho_1, rho_2 = p
    try:
        losses = evaluate_loss(
            variant,
            rho_1, rho_2, k,
            r_s1, v_s1, t1, attributable_1, ca_1,
            r_s2, v_s2, t2, attributable_2, ca_2,
            mu=mu, eps=eps,
        )
    except LambertNoSolution:
        return _INFEASIBLE_PENALTY
    return min(losses)


def optimize_hypothesis(
    variant, k, delta_t,
    c_1, r_s1, v_s1, t1, rho_1_bounds, attributable_1, ca_1,
    c_2, r_s2, v_s2, t2, rho_2_bounds, attributable_2, ca_2,
    mu=MU, eps=1e-6, a_min=A_MIN, a_max=A_MAX, history=None,
):
    """Equation (30): minimize L(p, k) over p = (rho_1, rho_2) in C, for one k.

    Parameters
    ----------
    variant : str
        Which dataset chain is being scored, one of VARIANTS. Selects
        the loss: the noisy chain uses the covariance-weighted Eq. (18),
        the clean chain the unweighted residual (see
        src/lab/scoring/discriminator_loss.py).
    k : float
        Revolutions/path parameter (fixed for this search -- Eq. 30 is
        solved once per admissible k, not jointly over k).
    delta_t : float
        t2 - t1 (s), for the Eq. 31 initial guess.
    c_1, c_2 : float
        r_s_i . u_i (src/lab/hypothesis/range_bounds.py), for the Eq. 31 guess.
    r_s1, r_s2 : array-like, shape (3,)
        Site positions (km) at t1, t2.
    v_s1, v_s2 : array-like, shape (3,)
        Site velocities (km/s) at t1, t2.
    t1, t2 : float
        Observation epochs (s).
    rho_1_bounds, rho_2_bounds : (float, float)
        (rho_min_i, rho_max_i) -- the admissible region C (Eq. 22).
    attributable_1, attributable_2 : array-like, shape (4,)
        [alpha, alpha_dot, delta, delta_dot] per tracklet.
    ca_1, ca_2 : array-like, shape (4, 4)
        Attributable covariance per tracklet.
    history : list, optional
        If given, every objective evaluation L-BFGS-B makes is appended as
        (rho_1, rho_2, loss), in call order. The seed is the first entry.
        Used by src/lab/evaluation/validate_optimiser.py to check the search
        actually descended rather than stalling on its first step.

    Returns
    -------
    rho_1_opt, rho_2_opt : float
        The minimizing range hypothesis for this k.
    loss_opt : float
        L(p_opt, k).
    result : scipy.optimize.OptimizeResult
        The full L-BFGS-B result, for inspecting convergence.
    """
    rho_1_0, rho_2_0 = circular_orbit_initial_guess(
        k, delta_t, c_1, r_s1, c_2, r_s2, mu, a_min, a_max,
    )
    rho_1_0 = float(np.clip(rho_1_0, *rho_1_bounds))
    rho_2_0 = float(np.clip(rho_2_0, *rho_2_bounds))

    objective = _objective
    if history is not None:
        def objective(p, *args):
            value = _objective(p, *args)
            history.append((float(p[0]), float(p[1]), float(value)))
            return value

    result = minimize(
        objective,
        x0=[rho_1_0, rho_2_0],
        args=(variant, k, r_s1, v_s1, t1, attributable_1, ca_1, r_s2, v_s2, t2, attributable_2, ca_2, mu, eps),
        method="L-BFGS-B",
        bounds=[rho_1_bounds, rho_2_bounds],
        options=_MINIMIZER_OPTIONS,
    )

    rho_1_opt, rho_2_opt = result.x
    return float(rho_1_opt), float(rho_2_opt), float(result.fun), result


def best_k_hypothesis(
    variant, k_min, k_max, delta_t,
    c_1, r_s1, v_s1, t1, rho_1_bounds, attributable_1, ca_1,
    c_2, r_s2, v_s2, t2, rho_2_bounds, attributable_2, ca_2,
    mu=MU, eps=1e-6, a_min=A_MIN, a_max=A_MAX,
):
    """Equation (30) over the full k grid: the paper's actual association test.

    Runs optimize_hypothesis for every k in {k_min, k_min+0.5, ..., k_max}
    and returns the one with the smallest min-loss -- the overall best-fit
    orbit hypothesis for this tracklet pair, independent of k.

    Returns
    -------
    k_opt : float
    rho_1_opt, rho_2_opt : float
    loss_opt : float
    """
    best = None
    for k in k_grid(k_min, k_max):
        rho_1_opt, rho_2_opt, loss_opt, _ = optimize_hypothesis(
            variant, k, delta_t,
            c_1, r_s1, v_s1, t1, rho_1_bounds, attributable_1, ca_1,
            c_2, r_s2, v_s2, t2, rho_2_bounds, attributable_2, ca_2,
            mu=mu, eps=eps, a_min=a_min, a_max=a_max,
        )
        if best is None or loss_opt < best[3]:
            best = (k, rho_1_opt, rho_2_opt, loss_opt)

    return best


def run_association_test(variant):
    """Run the full Eq. (30) all-vs-all association test for one chain.

    Returns
    -------
    true_losses, false_losses : list of float
        Optimized L* for the same-object and different-object pairs, kept
        separate so the caller can report the separation between them.
    """
    # --- Demo: run the full Eq. (30) association test over every tracklet
    # pair, and check the result against ground truth (object_id equality)
    # -- the first end-to-end test of whether this method actually works,
    # now that every stage from raw measurements to a real (not
    # placeholder-range) optimized loss exists. ---
    # The seed must span the same population the range bounds do, or it
    # lands outside [rho_min, rho_max] and gets clipped to a box corner.
    a_min, a_max, _ = population_for(variant)

    data = load_data(variant)

    with np.load(RANGE_BOUNDS_FILES[variant]) as archive:
        rb_object_ids = archive["object_id"]
        rb_tracklet_ids = archive["tracklet_id"]
        t0_values = archive["t0"]
        r_s_i = archive["r_s_i_km"]
        c_i = archive["c_i_km"]
        rho_min_i = archive["rho_min_i_km"]
        rho_max_i = archive["rho_max_i_km"]

    with np.load(K_BOUNDS_FILES[variant]) as archive:
        pair_object_id_1 = archive["object_id_1"]
        pair_tracklet_id_1 = archive["tracklet_id_1"]
        pair_object_id_2 = archive["object_id_2"]
        pair_tracklet_id_2 = archive["tracklet_id_2"]
        pair_k_min = archive["k_min"]
        pair_k_max = archive["k_max"]

    with np.load(ATTRIBUTABLES_FILES[variant]) as archive:
        attr_object_ids = archive["object_id"]
        attr_tracklet_ids = archive["tracklet_id"]
        attributables = archive["attributable"]
        covariances = archive["Ca"]

    from lab.hypothesis.range_bounds import v_s_i_at_epoch

    results = []
    for idx in range(len(pair_object_id_1)):
        i = _lookup_tracklet(rb_object_ids, rb_tracklet_ids, pair_object_id_1[idx], pair_tracklet_id_1[idx])
        j = _lookup_tracklet(rb_object_ids, rb_tracklet_ids, pair_object_id_2[idx], pair_tracklet_id_2[idx])
        ai = _lookup_tracklet(attr_object_ids, attr_tracklet_ids, rb_object_ids[i], rb_tracklet_ids[i])
        aj = _lookup_tracklet(attr_object_ids, attr_tracklet_ids, rb_object_ids[j], rb_tracklet_ids[j])

        delta_t = abs(t0_values[j] - t0_values[i])
        v_s_1 = v_s_i_at_epoch(data, rb_object_ids[i], rb_tracklet_ids[i], t0_values[i])
        v_s_2 = v_s_i_at_epoch(data, rb_object_ids[j], rb_tracklet_ids[j], t0_values[j])

        k_opt, rho_1_opt, rho_2_opt, loss_opt = best_k_hypothesis(
            variant, pair_k_min[idx], pair_k_max[idx], delta_t,
            c_i[i], r_s_i[i], v_s_1, t0_values[i], (rho_min_i[i], rho_max_i[i]), attributables[ai], covariances[ai],
            c_i[j], r_s_i[j], v_s_2, t0_values[j], (rho_min_i[j], rho_max_i[j]), attributables[aj], covariances[aj],
            a_min=a_min, a_max=a_max,
        )

        is_true_pair = int(rb_object_ids[i]) == int(rb_object_ids[j])
        results.append((
            int(rb_object_ids[i]), int(rb_tracklet_ids[i]),
            int(rb_object_ids[j]), int(rb_tracklet_ids[j]),
            is_true_pair, k_opt, rho_1_opt, rho_2_opt, loss_opt,
        ))

        print(
            f"Object {int(rb_object_ids[i])} tracklet {int(rb_tracklet_ids[i])} <-> "
            f"Object {int(rb_object_ids[j])} tracklet {int(rb_tracklet_ids[j])} "
            f"[{'TRUE' if is_true_pair else 'false'}]: "
            f"k*={k_opt}, rho*=({rho_1_opt:.1f}, {rho_2_opt:.1f}) km, L*={loss_opt:.6e}"
        )

    object_1, tracklet_1, object_2, tracklet_2, is_true, k, rho_1, rho_2, loss = (
        np.array(column) for column in zip(*results)
    )
    np.savez(
        ASSOCIATION_FILES[variant],
        object_id_1=object_1, tracklet_id_1=tracklet_1,
        object_id_2=object_2, tracklet_id_2=tracklet_2,
        is_true=is_true, k=k, rho_1_km=rho_1, rho_2_km=rho_2, loss=loss,
    )

    true_losses = loss[is_true].tolist()
    false_losses = loss[~is_true].tolist()

    print()
    print(f"True pairs:  n={len(true_losses)}, min L* range "
          f"[{min(true_losses):.3e}, {max(true_losses):.3e}]" if true_losses else "True pairs: none")
    print(f"False pairs: n={len(false_losses)}, min L* range "
          f"[{min(false_losses):.3e}, {max(false_losses):.3e}]" if false_losses else "False pairs: none")

    return true_losses, false_losses


if __name__ == "__main__":
    # Both chains are scored end to end. Their L* values are NOT comparable
    # across chains -- the clean chain's unweighted loss is in rad^2/s^2
    # and the noisy chain's weighted loss is dimensionless -- so each is
    # judged on its own true-vs-false separation.
    for variant in VARIANTS:
        if not (
            data_file(variant).exists()
            and ATTRIBUTABLES_FILES[variant].exists()
            and RANGE_BOUNDS_FILES[variant].exists()
            and K_BOUNDS_FILES[variant].exists()
        ):
            print(f"Skipping {variant}: inputs missing")
            continue

        print(f"--- {variant} ({LOSS_FUNCTIONS[variant].__name__}) ---")
        run_association_test(variant)
        print()
