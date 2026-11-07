# Testing

One suite per stage, in `test/`, each run as a script:

```bash
uv run python test/test_lambert.py          # one suite
for f in test/test_*.py; do uv run python "$f"; done   # all, about 2 minutes
```

Every suite generates its own data into a temporary folder, so none needs `data/`
or a run. Suites for stages 2–8 use `test/pipeline_fixture.py`. It calls the
current generator, `generate_object()`, then runs stages 2–4 in-process. Changing
the generator therefore re-tests every later stage against what it now writes.

How a stage is tested depends on what it contains:

- **Formula stages** (2, 3, 4, 6) evaluate a closed-form equation. They are tested
  with **simple cases** answered by hand, **edge cases**, a **defining property**
  where one exists, and a check on generated data.
- **Algorithm stages** (5, 7, 8) wrap a library black box. Each is tested for what
  the black box must do, for what the project's wrapper changes, and against the
  truth on generated data.

The truth comes from the clean chain's satellite columns. Only tests and metrics
read them.

| Stage | Suite | Tests |
|---|---|---|
| 1 Data | `test_gcrs_data.py`, `test_data_validation.py`, `test_station_velocity.py` | 53 |
| 1→2 reader | `test_dataset.py` | 12 |
| 2 Observation | `test_observation.py` | 16 |
| 3–4 Hypothesis | `test_hypothesis.py` | 25 |
| 5 Lambert | `test_lambert.py` | 18 |
| 5–6 Model | `test_model.py` | 13 |
| 7 Scoring | `test_discriminator_loss.py` | 15 |
| 8 Search | `test_search.py` | 13 |
| Metrics | `test_metrics.py` | 6 |

## Formula stages

**Reader, `dataset.py`.** The named columns match the generator's constants and
the column names it records in the settings file. `load_data` refuses an old
17-column file. `tracklet_rows` sorts by time and keeps objects apart. Stage paths
follow `DATA_DIR` when it is reassigned after import. Every module's `mu` is the
generator's.

**Stage 2, the attributable fit.** This is linear least squares,
`angle(t) = angle(t0) + rate·(t − t0)`.
- *Simple:* an exact line is recovered exactly, and a constant angle gives zero
  rate. For the generator's sampling of 5 points 20 s apart, the covariance is
  `σ²/5 + σ_sys²` for the angle and `σ²/4000` for the rate, with zero
  angle–rate covariance.
- *Edge:* an RA arc that crosses 2π, shuffled rows, two points (the minimum),
  one point (raises), uneven spacing, and zero noise (the zero matrix).
- *Data:* clean rates equal the true topocentric rates to 1e-9 rad/s. Because
  the fit is linear, noisy minus clean is exactly the fit of the noise, so
  `dᵀ Ca⁻¹ d` follows χ²(4) with mean 4. The 0.7″ and 2.0″ rate errors are
  in exact proportion, because the generator shares one set of noise draws
  across levels.

The clean fitted angle differs from the true angle by up to about 0.2″, which is
the arc's curvature. The rate is unaffected, because symmetric sampling cancels
the quadratic term.

**Stage 3, range bounds (Eqs. 23–28).**
- *Simple:* a site at the geocentre gives `ρ = |r|`. A zenith line of sight
  gives `|r| − |r_s|`. A horizontal one gives Pythagoras.
- *Property:* `|r_s + ρ_min u| = r_min` and `|r_s + ρ_max u| = r_max` over 200
  random geometries.
- *Edge:* `a_min = a_max` with `e = 0` collapses the bounds, larger `e` widens
  both, and interpolation is checked at and between samples.
- *Data:* every true range is inside its bounds in every chain, including the
  eccentric test case `'b'`.

**Stage 4, k bounds (Eq. 29).**
- *Simple:* the GEO period is a sidereal day, and 0.25, 0.75, 1.25 and 1.75
  periods give k = 0, 0.5, 1 and 1.5.
- *Edge:* dt = 0, dt exactly half a period, negative dt, and the paper's
  `k_min`/`k_max` labels being the wrong way round.
- *Data:* the true k of every same-object pair is in its grid, and every one of
  the `C(T,2)` pairs is kept, since per-object start offsets mean no two
  tracklets share an epoch.

**Stage 6, modelled rates (Eq. 17).**
- *Simple:* radial motion gives zero rates, and crossing motion gives `v/r`.
- *Property:* the rates match finite differences, and scaling position and
  velocity together leaves them unchanged.
- *Edge:* the rates are continuous across RA = 2π, and the RA rate is undefined
  at the pole.
- *Data:* at the true state, the modelled rates equal stage 2's clean rates.
  Without the station velocity they miss by more than 1″/s.

## Stage 5: Lambert's problem

**Black box:** `hapsira.core.iod.izzo(k, r1, r2, tof, M, prograde, lowpath, numiter, rtol)`,
Izzo's (2015) solver, numba-jitted, pinned at hapsira 0.18.0. It takes a
revolution count `M` and two flags, and returns one `(v1, v2)` or raises.

**What `lambert.py` changes.** It does not modify any velocity. It changes how
the solver is called and how results are chosen:

1. **It decodes the paper's `k`.** `M = floor(k)`, and the fractional part (0
   or 0.5) selects the short-way or long-way transfer.
2. **It does not trust hapsira's `prograde` flag to mean "short way".** That flag
   is measured against the fixed z axis. With r2 60° clockwise of r1,
   `prograde=True` returns the 300° long way. So the wrapper calls every
   `(prograde, lowpath)` combination and keeps the solutions whose actual swept
   angle matches `k`. It recomputes that angle from `r1`, `r2` and the
   solution's own `h = r1 × v1`.
3. **It returns both multi-revolution branches.** For `M ≥ 1` the low-path and
   high-path orbits are both valid, and Search scores each.
4. **It uses one exception type.** hapsira's bare `assert tof > 0`, its
   collinear-vector `ValueError`, and an empty result all become
   `LambertNoSolution`, which Search turns into a penalty.

**Tests.** Against the black box's contract: Curtis Example 5.2, the only
external reference; recovery of known orbits; and every candidate flying from
r1 to r2 in tof under both `two_body` and an independent scipy integration.
Against the wrapper's changes: candidates are bit-identical to raw hapsira
output, the hapsira flag behaviour above is pinned, and the short way is
returned in both rotation senses. The suite also checks k decoding, the
low/high-path pair, and 180° and infeasible cases. On generated data, in
`test_model.py`: at the true ranges and k, `build_orbit` recovers the true
velocity to better than 1e-8 km/s (3e-12 measured), and a wrong k never does.

## Stage 7: the discriminator loss

**There is no library black box.** The loss is this project's own code. It
calls stage 5 (hapsira) and stage 6 as its forward model, and estimates the
Eq. (19) Jacobian by central finite differences.

**What the paper prints, and what the code does:**

| | Paper as printed | Code | Why |
|---|---|---|---|
| Eq. (18) | `rᵀ (C_ż + C_ẑ̇) r`, no inverse | `rᵀ (C_ż + C_ẑ̇)⁻¹ r` via `solve` | The paper's own Eq. (21)/(13) likelihood and its χ² claim require the inverse |
| Eq. (19) | `Jᵀ C_z J` | `J C_z Jᵀ` | Standard propagation for `J = ∂ẑ̇/∂z` |
| Clean chain | — | `‖r‖²` in arcsec²/s² | `Ca = 0` makes the weighted form singular. In rad²/s² the values sit below L-BFGS-B's `ftol` |
| Several Lambert candidates | not covered | one loss each; Search takes the minimum | Multi-revolution transfers have two |

**Tests.** The algebra holds at any feasible point:
- scaling `Ca` by s scales L by 1/s;
- `C = c·I` reduces L to `‖r‖²/c`;
- an independent `solve()` recomputation agrees;
- the unweighted loss is `‖r‖²·206265²`;
- zero `Ca` raises `LinAlgError`;
- the dispatch by chain is correct.

For Eq. (19), a Monte Carlo covariance of `ẑ̇` over sampled z matches
`J C_z Jᵀ`, and the printed `Jᵀ C_z J` does not. On generated data at the
true ranges and k, the clean loss is below 1e-6 (1.5e-10 measured), the noisy
loss averages close to 4 as χ²(4) should, and moving 300 km off the truth
raises the loss.

## Stage 8: Search

**Black box:** `scipy.optimize.minimize(method="L-BFGS-B")`, a quasi-Newton
method with a finite-difference gradient. This is what the paper's Section 5.2
specifies.

**What `optimise_hypothesis.py` changes:**
- The gradient step is `eps = 1 km`, not scipy's default 1e-8, with tighter
  `ftol`/`gtol` and `maxiter = 500`. With the default, the gradient is rounding
  noise and the search stops at the seed.
- The Eq. (31) seed uses `a(k)` from the middle of the admissible revolution
  bin `[k, k + 0.5)`, not from `k` itself, where the paper's formula is
  undefined at k = 0.
- An infeasible trial point scores 1e12 instead of raising.
- Of several Lambert candidates, the lowest loss is used.

**Tests.**
- *Seed:* the bin midpoint, inside the population, away from 0°/180°
  transfers, and on the sphere of radius `a(k)`.
- *Edge:* dt = 0 and an unreachable k raise, and an infeasible hypothesis
  scores the penalty.
- *Data, 4 objects:* at the true k, clean searches reach the true ranges.
  Noisy searches score no worse than the truth, unless the straight line to
  the truth crosses a barrier. Over the k grid, Search picks the true k for
  at least 37 of 40 true pairs.

Two limits are measured and bounded rather than hidden. A flat penalty
plateau gives zero gradient, and some multi-revolution landscapes have a second
basin.
