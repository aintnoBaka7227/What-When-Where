# Pipeline Reference

Every stage, in order, with its exact inputs and outputs. For orientation and
project structure, see [README.md](README.md).

## Data flow

```
data/         gcrs_data_generation.py                      [1]
                     |
       generated_data_gcrs_<chain>.txt
                     |
observation/  create_attributes_vector.py                  [2]
                     |
       generated_attributables_<chain>.npz
              |                        |
hypothesis/   range_bounds.py          k_bounds.py         [3] [4]
              |                        |
       range_bounds_{...}.npz   k_bounds_{...}.npz
              \________________________/
                     |
model/        build_orbit.py (via lambert.py)              [5]
                     |
model/        model_rates.py                               [6]
                     |
scoring/      discriminator_loss.py                        [7]
                     |
search/       optimise_hypothesis.py                       [8]
                     |
              accept / reject per pair (not built)
```

Code paths are relative to `src/lab/`. Every generated file is in `data/`.

Stages 1–4 and 8 write files. Stage 5 prints a demonstration from its `__main__`
block, and stages 6 and 7 are pure libraries.

Through `experiment/run/run_<stage>.py`, every stage saves a result in its own
`results/runs/<hash>/output/`. Those scripts also save results for stages 5–7:
`model_{variant}.npz` from `run_model.py` and `surface_*.npz` from
`run_scoring.py`. A stage script run by hand uses `data/`.

Every stage processes all chains in one invocation (`clean`, `noisy_0.7as`,
`noisy_1.2as`, `noisy_2.0as`, defined in `src/lab/data/dataset.py`) and skips a
chain whose inputs are missing.

## Shared conventions

**The `data` matrix** written by stage 1 and read by stages 2, 3 and 6–8 has
18 columns:

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
| 17 | observation ID | — |

Columns 6–7 are the **only** columns that differ between the clean and noisy
files. Columns 0–5 are ground truth and are never read by the association
method — they exist to score it.

**The attributable** is `[alpha, alpha_dot, delta, delta_dot]` at a reference
epoch `t0`, in that order, everywhere in the project.

**Paths** are owned by `src/lab/hypothesis/range_bounds.py`, which defines `VARIANTS`
and the `GENERATED_DATA_FILES` / `ATTRIBUTABLES_FILES` / `PRECISION_OUTPUT_FILES`
/ `READABLE_OUTPUT_FILES` dictionaries. Stages 4–8 import these rather than
constructing paths. (`src/lab/observation/create_attributes_vector.py` is the one
exception — it holds its own copy, so that stage 2 does not have to import from
stage 3.)

---

## [1] `src/lab/data/gcrs_data_generation.py`

Generates one synthetic near-GEO object per run and appends it to every chain.

**Input.** Its own prior output, if present. The data files and settings record
must agree, or it raises before writing.

**Output.**

| File | Contents |
|---|---|
| `generated_data_gcrs_clean.txt` | exact angles |
| `generated_data_gcrs_noisy_{0.7,1.2,2.0}as.txt` | angles + measurement error at that level |
| `generated_data_gcrs_settings.json` | generation settings, plus each object's drawn orbit, stations and biases |

The text files are canonical: values are written at full precision and read back
exactly. `src/lab/data/dataset.py` reads them, and supplies each chain's
uncertainties and the orbit population from the settings record.

**What it does.** See [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) for the design.
Each object is seeded with its `object_id`, so object *n* is always identical.

---

## [2] `src/lab/observation/create_attributes_vector.py` + `fit_attributes.py`

Compresses each tracklet's RA/Dec series into a single attributable.

**Input.** `generated_data_gcrs_{variant}.txt` — columns 6, 7, 14, 15, 16,
plus the two uncertainty scalars.

**Output.** `data/generated_attributables_{variant}.npz` (+ `.txt`):

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

## [3] `src/lab/hypothesis/range_bounds.py`

Bounds the topocentric range for each tracklet — half the admissible region.

**Input.** `generated_attributables_{variant}.npz` (for `alpha`, `delta`, `t0`)
and `generated_data_gcrs_{variant}.txt` (for site position, interpolated to `t0`).

**Output.** `data/range_bounds_{variant}.npz` (+ `.txt`):

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

## [4] `src/lab/hypothesis/k_bounds.py`

Bounds the revolution/path parameter `k` for each tracklet **pair** — the other
half of the admissible region.

**Input.** `generated_attributables_{variant}.npz` — only `t0` and the IDs.

**Output.** `data/k_bounds_{variant}.npz` (+ `.txt`):

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

Because `k` depends only on `t0`, which the chains share exactly, their files
come out **identical**. They are still written separately so
each chain stays self-contained.

---

## [5] `src/lab/model/build_orbit.py` (+ `src/lab/model/lambert.py`)

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

`lambert.py` has no pipeline dependencies. `test/test_lambert.py` checks it against both Curtis's
worked example and a round-trip through this project's own `two_body`.

---

## [6] `src/lab/model/model_rates.py`

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

**Self-check.** `test/test_discriminator_loss.py` validates the analytic derivative against finite
differences before running the demo.

---

## [7] `src/lab/scoring/discriminator_loss.py`

Scores how badly a hypothesis's predicted rates disagree with the measured ones.
**This is where the chains diverge.**

**Input.** All four upstream `.npz` files. Writes nothing.

**Two loss functions:**

| Function | Used by | Formula |
|---|---|---|
| `discriminator_loss` | noisy chains | `residualᵀ (C_z_dot + C_z_hat_dot)⁻¹ residual` |
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

## [8] `src/lab/search/optimise_hypothesis.py`

Searches the admissible region for the best-fitting orbit. **The actual
association test.**

**Input.** All four upstream `.npz` files.

**Output.** `association_{variant}.npz`, one row per pair:

| Key | Meaning |
|---|---|
| `object_id_1`, `tracklet_id_1`, `object_id_2`, `tracklet_id_2` | the pair |
| `is_true` | ground truth: same object |
| `k`, `rho_1_km`, `rho_2_km` | the best hypothesis |
| `loss` | its loss, `L*` |

It also writes `search_trace_{variant}.npz`, one row per `(pair, k)` search, with
the seed loss, final loss, iterations and loss evaluations.

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

All eight stages run end to end on all four chains. On a 10-object GCRS dataset
(50 tracklets, 1,225 pairs), with inclination 0–20°, run `cdfe67c70d10`:

| chain | best false pair | true pairs at or above it | best threshold | missed true | accepted false |
|---|---|---|---|---|---|
| clean | 9.68e−02 | 15 of 100 | 1.6e−05 | 15 | 0 |
| noisy_0.7as | 8.05e+02 | 15 of 100 | 10.9 | 15 | 0 |
| noisy_1.2as | 2.79e+02 | 15 of 100 | 10.6 | 15 | 0 |
| noisy_2.0as | 1.03e+02 | 15 of 100 | 10.4 | 15 | 0 |

The thresholds were chosen on this same data, so they are not validated.

The 15 failures are a Search problem. All 100 true pairs score at most 2.5e−10 at
their true ranges and `k` on the clean chain. In 5 failures, the only `k` tried
returns the 1e+12 penalty; in the other 10, Search chooses the wrong `k`. Only 1 of
the 15 is within 5° of 0°/180°, so degeneracy is not the main cause.

## Metrics

`src/lab/evaluation/metrics.py` judges each stage against the simulated truth.
`experiment/run/` saves the results as `metrics.json` in each run's folder.

| Stage | Metric | Target |
|---|---|---|
| Hypothesis | True range inside the bounds | 100% |
| Hypothesis | True `k` inside the grid | 100% |
| Scoring | Loss at the true ranges and `k` | Clean near 0; noisy mean near 4 |
| Search | Correct `k` on true pairs | 100% |
| Search | Searches that start on the penalty | 0 |
| Search | Iterations and loss evaluations | Paper: 2–3 steps, capped at 10; about 20 evaluations |
| Evaluation | FN and FP at the best threshold and at T = 1–20 | Low on both; noisy chains only for T = 1–20 |
| Evaluation | Separation gap: best false ÷ worst true | Above 1 |

Read correct `k` together with `L*`: a pair whose only `k` hits the penalty counts
as correct but is still missed. The best threshold is chosen on the pairs it is
scored on, so its rates are optimistic.

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

## Gotchas

- **Stage 1 appends.** Every run adds one object. Run scripts start each
  dataset empty; delete
  `data/generated_data_gcrs_*` to reset stages run by hand.
- **One object means no test.** A single object gives only same-object pairs, so
  stage 8 reports `False pairs: none` and proves nothing.
- **Stage 2 onward `.txt` files are for reading.** Their `.npz` files are canonical.
  Stage 1's `.txt` files are canonical and exact.
- **Clean data cannot use the weighted loss.** It raises
  `LinAlgError: Singular matrix`. This is handled automatically by
  `LOSS_FUNCTIONS`, but matters if you call the loss directly.
- **RA uncertainty is not divided by cos(δ).** Generator and fitter make the
  same assumption, so they are self-consistent, but this diverges slightly from
  standard astrometric convention.
- **Stage 8 cost grows quadratically** in tracklet count. Four objects is ~90 s.
