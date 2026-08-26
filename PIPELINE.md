# Pipeline Reference

Every stage, in order, with its exact inputs and outputs. For orientation and
project structure, see [README.md](README.md).

## Data flow

```
                 simulation/new_data_gen.py                    [1]
                            |
              generated_data_{clean,noisy}.npz
                            |
          observation/create_attributes_vector.py              [2]
                            |
          generated_attributables_{clean,noisy}.npz
                    |                    |
    hypothesis/range_bounds.py    hypothesis/k_bounds.py     [3] [4]
                    |                    |
      range_bounds_{...}.npz     k_bounds_{...}.npz
                    \____________________/
                            |
                 orbit/build_orbit.py                     [5]
                     (via orbit/lambert.py)
                            |
                  modelled/model_rates.py                      [6]
                            |
             discriminator/discriminator_loss.py               [7]
                            |
            optimisation/optimise_hypothesis.py                [8]
                            |
                   accept / reject per pair
```

Stages 1–4 write files. Stages 5–8 are libraries whose `__main__` blocks print
demonstrations; they persist nothing.

Every stage processes both the `clean` and `noisy` variants in one invocation
and skips a variant whose inputs are missing.

## Shared conventions

**The `data` matrix** written by stage 1 and read by stages 3 and 6–8 has 17
columns:

| Cols | Meaning | Units |
|---|---|---|
| 0–2 | true satellite position | km |
| 3–5 | true satellite velocity | km/s |
| 6 | measured right ascension `alpha` | rad |
| 7 | measured declination `delta` | rad |
| 8–10 | observing site position | km |
| 11–13 | observing site velocity | km/s |
| 14 | observation time since epoch | s |
| 15 | tracklet ID | — |
| 16 | object ID | — |

Columns 6–7 are the **only** columns that differ between the clean and noisy
files. Columns 0–5 are ground truth and are never read by the association
method — they exist to score it.

**The attributable** is `[alpha, alpha_dot, delta, delta_dot]` at a reference
epoch `t0`, in that order, everywhere in the project.

**Paths** are owned by `hypothesis/range_bounds.py`, which defines `VARIANTS`
and the `GENERATED_DATA_FILES` / `ATTRIBUTABLES_FILES` / `PRECISION_OUTPUT_FILES`
/ `READABLE_OUTPUT_FILES` dictionaries. Stages 4–8 import these rather than
constructing paths. (`observation/create_attributes_vector.py` is the one
exception — it holds its own copy, so that stage 2 does not have to import from
stage 3.)

---

## [1] `simulation/new_data_gen.py`

Generates one synthetic near-GEO object per run and appends it to both datasets.

**Input.** Its own prior output, if present. Both `.npz` files must exist or
neither — the mismatch is caught and raised, since the two must stay in step.

**Output.**

| File | Contents |
|---|---|
| `simulation/generated_data_clean.npz` | exact angles |
| `simulation/generated_data_noisy.npz` | angles + measurement error |
| `..._clean.txt`, `..._noisy.txt` | readable copies (lossy, `%f`) |

Each `.npz` holds:

| Key | Shape | Meaning |
|---|---|---|
| `data` | (N, 17) | the matrix above |
| `true_angles` | (N, 5) | `[alpha, delta, t, tracklet_id, object_id]`, noise-free |
| `relative_uncertainty_arcsec` | scalar | 0.0 clean, 0.7 noisy |
| `systematic_uncertainty_arcsec` | scalar | 0.0 clean, 5.0 noisy |
| `random_seed` | scalar | the object ID this run generated |

**What it does.** Draws a near-GEO orbit (`a ∈ [38000, 45000] km`,
`e ≤ 0.01` for the default test case `'a'`), then for each of 3 tracklets
places a ground station beneath the sub-satellite point, propagates the orbit
with `two_body`, and converts the site-relative position to RA/Dec. Tracklets
are 5 points 30 s apart, separated by 12 h.

Noise is applied per tracklet as one constant RA bias and one constant Dec bias
(the systematic term), plus an independent draw per observation (the relative
term).

**Notes.** Appends one object per run, seeded by `object_id`, so runs are
reproducible and repeatable. Station placement guarantees visibility by
construction — every object is above the horizon at every epoch. Test cases
`'b'`–`'d'` raise eccentricity beyond what stage 3's `E_MAX` assumes; only
`'a'` is consistent with the rest of the pipeline as configured.

---

## [2] `observation/create_attributes_vector.py` + `fit_attributes.py`

Compresses each tracklet's RA/Dec series into a single attributable.

**Input.** `simulation/generated_data_{variant}.npz` — columns 6, 7, 14, 15, 16,
plus the two uncertainty scalars.

**Output.** `observation/generated_attributables_{variant}.npz` (+ `.txt`):

| Key | Shape | Meaning |
|---|---|---|
| `object_id`, `tracklet_id` | (T,) | identity |
| `t0` | (T,) | reference epoch, the mean observation time |
| `attributable` | (T, 4) | `[alpha, alpha_dot, delta, delta_dot]` |
| `Ca` | (T, 4, 4) | covariance, same ordering |

**What it does.** `fit_attributes()` unwraps RA, centres time on `t0`, and
least-squares fits `angle(t) = angle(t0) + rate·(t − t0)` to RA and Dec
separately. The covariance is `sigma_relative² (XᵀX)⁻¹`, with `sigma_systematic²`
added to the two angle variances only — a constant whole-tracklet offset moves
the fitted angle but not the fitted rate.

**Critical consequence.** The clean dataset carries zero uncertainties, so its
`Ca` is the **zero matrix**. This is correct and deliberate, but it means the
clean chain cannot use any loss that inverts `Ca`. See stage 7.

---

## [3] `hypothesis/range_bounds.py`

Bounds the topocentric range for each tracklet — half the admissible region.

**Input.** `generated_attributables_{variant}.npz` (for `alpha`, `delta`, `t0`)
and `generated_data_{variant}.npz` (for site position, interpolated to `t0`).

**Output.** `hypothesis/range_bounds_{variant}.npz` (+ `.txt`):

| Key | Shape | Meaning |
|---|---|---|
| `object_id`, `tracklet_id`, `t0` | (T,) | identity and epoch |
| `r_s_i_km` | (T, 3) | site position at `t0` |
| `u_i` | (T, 3) | line-of-sight unit vector from `alpha`, `delta` |
| `c_i_km` | (T,) | `r_s · u` |
| `rho_min_i_km`, `rho_max_i_km` | (T,) | the range bounds |
| `a_min_km`, `a_max_km`, `e_max`, `r_min_km`, `r_max_km` | scalars | the population assumptions used |

**Equations.** (23), (24) for `‖r_min‖ = a_min(1 − e_max)` and
`‖r_max‖ = a_max(1 + e_max)`; (26)–(28) for
`rho = −c_i + sqrt(c_i² + ‖r‖² − ‖r_s‖²)`.

**Assumptions.** `A_MIN = 38000`, `A_MAX = 45000`, `E_MAX = 0.01` — the paper's
near-GEO population, matching stage 1's test case `'a'`. Changing the generator's
orbit family means changing these too.

**Also provides** `r_s_i_at_epoch()` and `v_s_i_at_epoch()`, which interpolate
site position and velocity to `t0`. Stages 6–8 import them.

---

## [4] `hypothesis/k_bounds.py`

Bounds the revolution/path parameter `k` for each tracklet **pair** — the other
half of the admissible region.

**Input.** `generated_attributables_{variant}.npz` — only `t0` and the IDs.

**Output.** `hypothesis/k_bounds_{variant}.npz` (+ `.txt`):

| Key | Shape | Meaning |
|---|---|---|
| `object_id_1`, `tracklet_id_1`, `object_id_2`, `tracklet_id_2` | (P,) | the pair |
| `delta_t_s` | (P,) | \|t0₂ − t0₁\| |
| `k_min`, `k_max` | (P,) | admissible revolution range |
| `a_min_km`, `a_max_km` | scalars | assumptions used |

**Equation.** (29): `k = floor(2 Δt / P(a)) / 2`, evaluated at both `a_min` and
`a_max`. Because period grows with `a`, the raw value from `a_min` is the
*larger* one despite the paper's naming; this module takes `min()`/`max()`
rather than trusting the labels.

**Notes.** All unordered pairs are considered, matching the paper's all-vs-all
test — the method is not told which tracklets belong together. Pairs with
`delta_t == 0` are dropped, since Lambert needs two distinct epochs and all
objects share a start epoch. For T tracklets across O objects this gives fewer
than `C(T,2)` pairs.

Because `k` depends only on `t0`, which the two chains share exactly, the clean
and noisy files come out **identical**. They are still written separately so
each chain stays self-contained.

---

## [5] `orbit/build_orbit.py` (+ `orbit/lambert.py`)

Turns a range hypothesis into an actual orbit.

**Input.** `range_bounds_{variant}.npz` and `k_bounds_{variant}.npz`. Writes
nothing.

**Key function.**
`build_orbit(r_s_1, u_1, t1, rho_1, r_s_2, u_2, t2, rho_2, k)`
→ `(r_hat_1, r_hat_2, delta_t, candidates)`, where each candidate is a
`(v1, v2)` velocity pair.

**Equation.** (5): `r_hat = r_s + rho·u` — a hypothesised range plus a known
site position and line of sight gives a geocentric position. Two of those plus
a flight time is a Lambert boundary-value problem.

`lambert.py` provides `lambert_boundary_value(mu, r1, r2, tof, k)`, raising
`LambertNoSolution` for infeasible geometry. It decodes `k` into revolution
count and transfer direction. Multi-revolution cases can return **two**
candidate orbits; everything downstream carries the list and scores each.

`lambert.py` has no pipeline dependencies and self-tests against both Curtis's
worked example and a round-trip through this project's own `two_body`.

---

## [6] `modelled/model_rates.py`

Predicts what angular rates a candidate orbit would produce.

**Input.** All four upstream `.npz` files. Writes nothing.

**Key functions.** `angle_rate_from_state(rho_vec, rho_dot_vec)` differentiates
RA/Dec analytically:

```
alpha_dot = (px·vy − py·vx) / s²
delta_dot = (vz·s² − pz·(px·vx + py·vy)) / (r²·s)
```

`modeled_angular_rates(...)` → `z_hat_dot` = `(alpha_dot_1, delta_dot_1,
alpha_dot_2, delta_dot_2)`, Eq. (17).

**Self-check.** Its `__main__` validates the analytic derivative against finite
differences before running the demo.

---

## [7] `discriminator/discriminator_loss.py`

Scores how badly a hypothesis's predicted rates disagree with the measured ones.
**This is where the two chains diverge.**

**Input.** All four upstream `.npz` files. Writes nothing.

**Two loss functions:**

| Function | Used by | Formula |
|---|---|---|
| `discriminator_loss` | noisy | `residualᵀ (C_z_dot + C_z_hat_dot)⁻¹ residual` |
| `discriminator_loss_unweighted` | clean | `‖residual‖²`, in arcsec²/s² |

Both are built on `rate_residuals()`, which computes `z_dot − z_hat_dot` per
Lambert candidate, so the two cannot drift apart. `LOSS_FUNCTIONS` maps chain →
loss; `evaluate_loss(variant, ...)` dispatches with a uniform signature.

**Why two.** The clean chain's `Ca` is the zero matrix (stage 2), so the
weighted form would invert a singular matrix. With no covariance there is
nothing to weight by, and the plain squared residual is the only meaningful
score. Dropping the weighting also drops the Eq. (19) Jacobian — which exists
only to propagate `C_z` — making the clean loss far cheaper: one Lambert solve
per call instead of nine.

**Why arcsec.** Near-GEO rates are ~1e−5 rad/s, so squared residuals in
rad²/s² land near 1e−10 — below L-BFGS-B's default `ftol` of ~2.2e−9. The
optimiser would report convergence on its first step and never leave the
initial guess. Scoring in arcsec/s is a pure unit change that puts the same
landscape in a range the optimiser can search.

**The losses are not comparable across chains.** The clean loss is in
arcsec²/s²; the weighted loss is dimensionless. Compare clean to clean and
noisy to noisy.

**On the matrix inverse.** Eq. (18) as *printed* omits the inverse. The paper's
own derivation requires it: Eq. (21) defines the likelihood as a Gaussian whose
exponent (Eq. 13) is explicitly inverted, the text calls Eq. (18) that exponent,
and the chi-squared claim only holds for a quadratic form in the inverse
covariance. It is implemented with the inverse. That the unweighted variant
happens to match the printed form is a coincidence, not the reason for it.

---

## [8] `optimisation/optimise_hypothesis.py`

Searches the admissible region for the best-fitting orbit. **The actual
association test.**

**Input.** All four upstream `.npz` files. Writes nothing — prints results.

**Key functions.**

- `circular_orbit_initial_guess(k, delta_t, c_1, r_s1, c_2, r_s2)` — Eq. (31),
  the range a circular orbit with the right period would imply.
- `optimize_hypothesis(variant, k, delta_t, ...)` — Eq. (30), minimising over
  `p = (rho_1, rho_2)` within the bounds, for one `k`, via L-BFGS-B.
- `best_k_hypothesis(variant, k_min, k_max, ...)` — the above over the whole `k`
  grid, returning the overall best `(k, rho_1, rho_2, loss)`.
- `run_association_test(variant)` — all pairs, comparing the result against
  ground truth (`object_id` equality) and reporting true/false loss ranges.

`variant` is the first argument throughout so the loss choice reaches into the
objective function.

`LambertNoSolution` inside the objective returns a large finite penalty rather
than propagating, so the optimiser can walk out of infeasible regions.

---

## Current status

All eight stages run end to end on both chains. On a four-object dataset
(12 tracklets, 48 pairs), both chains separate true from false pairs completely:

| chain | true pairs | false pairs | margin |
|---|---|---|---|
| clean | 7.09e−09 – 3.32e−05 | 1.21e+02 – 2.39e+03 | 3.6e+06 × |
| noisy | 7.14e−02 – 7.54e+03 | 2.21e+06 – 4.38e+07 | 293 × |

Every same-object pair scores below every different-object pair, so a single
threshold classifies all 48 correctly in each chain.

## What's not built yet

- **Accept/reject gating.** `best_k_hypothesis()` finds the best hypothesis and
  its loss; nothing applies a threshold to turn that into a decision, or handles
  several `k` passing it (Section 5.2: "if more k values fulfill the
  condition[,] multiple candidates are created"). The separation above is what a
  threshold would be chosen against.
- **Adversarial simulation.** Each object is an independent draw from the same
  near-GEO population, not a *deliberately similar* orbit. The paper's hard case
  — and the reason it says false associations "cannot be completely eliminated"
  — is closely spaced objects on similar orbits. Reproducing Fig. 9 needs that
  scenario, not just more random objects.
- **Other noise cases.** The paper tests 0.7″, 1.2″ and 2.0″ relative error.
  Only 0.7″ is wired up. Adding the others means extending `VARIANTS` in
  `hypothesis/range_bounds.py` *and* `DATASETS` in
  `observation/create_attributes_vector.py`.

## Gotchas

- **Stage 1 appends.** Every run adds one object. Delete
  `simulation/generated_data_{clean,noisy}.{npz,txt}` to reset.
- **One object means no test.** A single object gives only same-object pairs, so
  stage 8 reports `False pairs: none` and proves nothing.
- **The `.txt` files are lossy.** RA/Dec are written with `%f` — six decimals is
  0.206″ of quantisation against 0.7″ of noise. Never reproduce results from
  them; the `.npz` is canonical.
- **Clean data cannot use the weighted loss.** It raises
  `LinAlgError: Singular matrix`. This is handled automatically by
  `LOSS_FUNCTIONS`, but matters if you call the loss directly.
- **RA uncertainty is not divided by cos(δ).** Generator and fitter make the
  same assumption, so they are self-consistent, but this diverges slightly from
  standard astrometric convention.
- **Stage 8 cost grows quadratically** in tracklet count. Four objects is ~90 s.
