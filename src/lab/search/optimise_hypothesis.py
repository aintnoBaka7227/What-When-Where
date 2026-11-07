"""Stage 8: minimise the loss over the admissible region, Eqs. (30)-(31).

    p_k = argmin_{p in C} L(p, k),   p = (rho_1, rho_2)            Eq. (30)
    rho_i* = -c_i + sqrt(c_i^2 + a(k)^2 - ||r_s_i||^2)           Eq. (31)

C is each rho_i's [rho_min_i, rho_max_i] (Eq. 22). The search starts at the
circular-orbit seed rho*. As printed, a(k) is undefined at k = 0 and k covers
a range of revolutions, [k, k + 0.5); a(k) here comes from the middle of the
part of that range the population allows (semi_major_axis_for_k).

With several Lambert candidates (M > 0) the objective takes the smallest
loss at each trial point.

The search is scipy's L-BFGS-B with a finite-difference gradient: the
quasi-Newton method with numerical derivatives of Section 5.2. Its step and
tolerances are set in _MINIMIZER_OPTIONS; the defaults stop early.
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

# km^3/s^2, as in the generator.
MU = GM_earth.value / 1_000_000_000.0

# Score for a trial point with no Lambert solution, so the search keeps
# running. Flat, so a search seeded inside an infeasible region stays there.
_INFEASIBLE_PENALTY = 1.0e12

# scipy's default gradient step (~1e-8) is lost in rounding on ~4e4 km ranges,
# so the search "converges" near the seed, far above the minimum (see
# evaluation/validate_optimiser.py). 1 km resolves the loss valley, which is
# ~500 km wide even near 180 deg transfers. ftol, gtol and maxiter match.
# Not the same eps as the Eq. (19) Jacobian's.
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
    """Semi-major axis for the Eq. (31) seed; raises if k is unreachable."""
    delta_t = abs(delta_t)

    if delta_t == 0.0:
        raise ValueError("Tracklet epochs must be different")

    period_a_min = 2.0 * np.pi * np.sqrt(a_min**3 / mu)
    period_a_max = 2.0 * np.pi * np.sqrt(a_max**3 / mu)

    # The part of [k, k + 0.5) that [a_min, a_max] allows.
    revolutions_min = delta_t / period_a_max
    revolutions_max = delta_t / period_a_min

    lower = max(k, revolutions_min)
    upper = min(k + 0.5, revolutions_max)

    if lower >= upper:
        raise ValueError(
            f"k={k} does not overlap [{a_min}, {a_max}] km"
        )

    # Its midpoint, k + 0.25 when all of it is allowed: a 90 or 270 deg
    # transfer, away from where Lambert degenerates. The old lower-edge seed
    # (near 0 deg) left about 1 in 6 true-k searches on the penalty.
    revolutions_star = 0.5 * (lower + upper)
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
    """Eq. (31): (rho_1*, rho_2*) for a circular orbit of radius a(k)."""
    a_k = semi_major_axis_for_k(k, delta_t, mu, a_min, a_max)

    r_s1_norm_sq = np.dot(r_s1, r_s1)
    r_s2_norm_sq = np.dot(r_s2, r_s2)

    # Clipped at 0 against rounding; this is only a seed.
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
    """Eq. (30) for one fixed k.

    variant selects the loss. Per tracklet: c = r_s . u, site position and
    velocity, epoch, (rho_min, rho_max) bounds, attributable and Ca.

    history, if given, collects every evaluation as (rho_1, rho_2, loss),
    seed first.

    Returns (rho_1_opt, rho_2_opt, loss_opt, scipy OptimizeResult).
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
    mu=MU, eps=1e-6, a_min=A_MIN, a_max=A_MAX, trace=None,
):
    """Eq. (30) over the whole k grid: (k_opt, rho_1_opt, rho_2_opt, loss_opt).

    trace, if given, collects (k, seed loss, final loss, iterations,
    evaluations) per k.
    """
    best = None
    for k in k_grid(k_min, k_max):
        history = []
        rho_1_opt, rho_2_opt, loss_opt, result = optimize_hypothesis(
            variant, k, delta_t,
            c_1, r_s1, v_s1, t1, rho_1_bounds, attributable_1, ca_1,
            c_2, r_s2, v_s2, t2, rho_2_bounds, attributable_2, ca_2,
            mu=mu, eps=eps, a_min=a_min, a_max=a_max, history=history,
        )
        if trace is not None:
            trace.append((k, history[0][2], loss_opt, result.nit, len(history)))
        if best is None or loss_opt < best[3]:
            best = (k, rho_1_opt, rho_2_opt, loss_opt)

    return best


def run_association_test(variant):
    """Search every pair of one chain; save the results and per-k trace.

    Returns (true_losses, false_losses), split by object_id equality.
    """
    # The seed must use the bounds' population, or it is clipped to a corner.
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
    searches = []
    for idx in range(len(pair_object_id_1)):
        i = _lookup_tracklet(rb_object_ids, rb_tracklet_ids, pair_object_id_1[idx], pair_tracklet_id_1[idx])
        j = _lookup_tracklet(rb_object_ids, rb_tracklet_ids, pair_object_id_2[idx], pair_tracklet_id_2[idx])
        ai = _lookup_tracklet(attr_object_ids, attr_tracklet_ids, rb_object_ids[i], rb_tracklet_ids[i])
        aj = _lookup_tracklet(attr_object_ids, attr_tracklet_ids, rb_object_ids[j], rb_tracklet_ids[j])

        delta_t = abs(t0_values[j] - t0_values[i])
        v_s_1 = v_s_i_at_epoch(data, rb_object_ids[i], rb_tracklet_ids[i], t0_values[i])
        v_s_2 = v_s_i_at_epoch(data, rb_object_ids[j], rb_tracklet_ids[j], t0_values[j])

        trace = []
        k_opt, rho_1_opt, rho_2_opt, loss_opt = best_k_hypothesis(
            variant, pair_k_min[idx], pair_k_max[idx], delta_t,
            c_i[i], r_s_i[i], v_s_1, t0_values[i], (rho_min_i[i], rho_max_i[i]), attributables[ai], covariances[ai],
            c_i[j], r_s_i[j], v_s_2, t0_values[j], (rho_min_i[j], rho_max_i[j]), attributables[aj], covariances[aj],
            a_min=a_min, a_max=a_max, trace=trace,
        )

        is_true_pair = int(rb_object_ids[i]) == int(rb_object_ids[j])
        searches.extend((idx, is_true_pair, *search) for search in trace)
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

    pair, search_is_true, search_k, seed_loss, final_loss, iterations, evaluations = (
        np.array(column) for column in zip(*searches)
    )
    np.savez(
        ASSOCIATION_FILES[variant].with_name(f"search_trace_{variant}.npz"),
        pair=pair, is_true=search_is_true, k=search_k, seed_loss=seed_loss,
        final_loss=final_loss, iterations=iterations, evaluations=evaluations,
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
    # L* is not comparable across chains: clean is in arcsec^2/s^2, noisy is
    # dimensionless.
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
