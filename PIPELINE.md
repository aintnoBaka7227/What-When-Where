# Tracklet Association Pipeline

This document describes the codebase as it stands today: what it's for, how the
pieces fit together, and how to run them. `README.md` documents only the
original synthetic-data script; this file covers everything built since.

## Purpose

This project implements (incrementally) the boundary-value tracklet
association method from:

> J.A. Siminski, O. Montenbruck, H. Fiedler, T. Schildknecht, "Short-arc
> tracklet association for geostationary objects," *Advances in Space
> Research* 53 (2014) 1184-1194. (`sources/siminski.pdf`)

**The problem:** an optical telescope observes a satellite for a minute or
two (a "tracklet": 3-5 right-ascension/declination measurements), then loses
it. Later, it (or another object) is observed again in a second tracklet. Do
the two tracklets belong to the same object? With only angles and angular
rates -- no range, no velocity -- this is underdetermined: each tracklet only
constrains 4 of an orbit's 6 degrees of freedom.

**The paper's approach:** hypothesize the unknown range (distance from
observer to object) at both tracklets' epochs, solve Lambert's problem (the
boundary-value orbit problem: given two positions and a flight time, find the
connecting two-body orbit) to get a candidate orbit, then check whether that
orbit's *predicted* angular rates match what was *actually measured*. Search
over the hypothesized ranges (and the number of orbital revolutions/transfer
direction, `k`) using gradient-based optimization to find the best-fitting
orbit; if its mismatch is small enough, accept the association.

This codebase builds that pipeline stage by stage, matching the paper's own
equation numbering throughout so the code and the paper can be read side by
side.

## Environment

- Python >= 3.11 (bumped from 3.9 because `astropy`'s numpy-2.0 support
  requires it -- see "Notes and gotchas" below)
- Dependency management via [`uv`](https://docs.astral.sh/uv/); `uv sync`
  installs everything from `pyproject.toml` / `uv.lock`
- Key dependencies: `numpy`, `scipy`, `numba` (JIT-compiles the hot-path
  orbit math), `astropy` (Earth constants, time/frame handling), `hapsira`
  (MIT-licensed fork of `poliastro`; supplies the Izzo-algorithm Lambert
  solver core)

## Pipeline stages

```
simulation/new_data_gen.py
        |  writes generated_data.npz  (truth state + noisy RA/Dec per measurement)
        v
observation/create_attributes_vector.py
        |  writes generated_attributables.npz  (fitted alpha, alpha_dot, delta, delta_dot per tracklet)
        v
orbit/range_bounds.py                    orbit/k_bounds.py
        |  per-tracklet rho bounds               |  per-tracklet-PAIR k (revolutions/path) grid
        |  writes range_bounds.npz                |  writes k_bounds.npz
        v                                          v
        +--------------------+-----------------+
                              v
hypothesis/build_hypothesis.py
        |  rho -> r (Eq. 5), then boundary-value Lambert solve
        v
dynamic/lambert.py
        |  lambert_boundary_value(mu, r1, r2, tof, k) -> candidate (v1, v2) orbits
        v
   [not yet built: discriminator loss (Eq. 18), BFGS optimizer (Eq. 30),
    accept/reject gating]
```

Every stage writes both a `.npz` (full precision, for the next stage to
consume) and a human-readable `.txt` (for inspection).

### 1. `simulation/new_data_gen.py` -- synthetic tracklets

Generates one synthetic near-GEO object (`test_case = 'a'`: semi-major axis
38,000-45,000 km, eccentricity 0-0.01), propagates it with a two-body
(`two_body`) Kepler propagator, and observes it from one synthetic ground
site across 3 tracklets (5 measurements each, 30 s apart, ~12 h gaps between
tracklets -- matching the paper's Section 6 setup: 3-5 measurements per
tracklet, 1-2 minute duration). Each measurement gets independent relative
noise (0.7 arcsec) plus a per-tracklet constant systematic offset (5 arcsec),
matching the paper's two-component error model (Section 6, citing Herzog
et al. 2013).

Also contains the general-purpose orbital-mechanics toolkit everything else
either uses or is patterned after: classical/equinoctial orbital element
conversions (`gcrs_to_coe`, `coe_to_gcrs`, `oe_to_gcrs`, `gcrs_to_oe`,
`coe_to_eq_oe`), the two-body propagator (`two_body`, Numba-jitted), and
`gcrs_to_radec`/`radec_to_point` for angle conversions.

**Run:** `uv run python simulation/new_data_gen.py`
**Writes:** `simulation/generated_data.npz`, `simulation/generated_data.txt`
**Reads:** nothing (entry point of the pipeline)

Random seed is fixed (`42`), so re-running regenerates *identical* data --
this is deliberate, so the rest of the pipeline can be re-verified
deterministically. The output files are fully overwritten each run, not
appended (see "Notes and gotchas").

### 2. `observation/fit_attributes.py` + `create_attributes_vector.py` -- attributables

Per Eq. 1, an attributable is `a = (alpha, alpha_dot, delta, delta_dot)`: a
tracklet's angular position and rate at one reference epoch, fit by linear
least squares from the raw RA/Dec measurements (`fit_attributes.py`), with a
propagated covariance that separately accounts for the relative
(point-to-point) and systematic (whole-tracklet) noise components.
`create_attributes_vector.py` runs this over every tracklet in the simulated
data.

**Run:** `uv run python observation/create_attributes_vector.py`
**Writes:** `observation/generated_attributables.npz` (`object_id`,
`tracklet_id`, `t0`, `attributable` = `[alpha, alpha_dot, delta, delta_dot]`,
`Ca` = 4x4 covariance), `generated_attributables.txt`
**Reads:** `simulation/generated_data.npz`

### 3. `orbit/range_bounds.py` -- admissible range per tracklet

Per Eqs. 22-28: given the population's assumed semi-major-axis interval
(`A_MIN`, `A_MAX`) and max eccentricity (`E_MAX`), and each tracklet's fitted
line-of-sight direction `u_i` (from `alpha`, `delta`) and site position
`r_s_i` (interpolated to the attributable epoch `t0`), computes the
topocentric range interval `[rho_min_i, rho_max_i]` -- how far along that
line of sight the object could plausibly be, given the assumed orbital
population.

**Run:** `uv run python orbit/range_bounds.py`
**Writes:** `orbit/range_bounds.npz` (`r_s_i_km`, `u_i`, `c_i_km`,
`rho_min_i_km`, `rho_max_i_km`, per tracklet), `range_bounds.txt`
**Reads:** `simulation/generated_data.npz`, `observation/generated_attributables.npz`

### 4. `orbit/k_bounds.py` -- admissible revolutions/path per tracklet PAIR

Per Eq. 29: for *every pair* of tracklets (all-vs-all, matching the paper's
Section 6.1 association test -- the pipeline doesn't get to assume which
tracklets belong together), computes the admissible range of `k`, the
revolutions/transfer-direction parameter: `k`'s integer part is the number of
full orbital revolutions between the two tracklets, and its fractional part
(`0` or `0.5`) selects the short-way (<180 deg) or long-way (>180 deg)
transfer. Bounds come from Kepler's third law applied at `a_min` and `a_max`
over the pair's time separation. `k_grid()` enumerates the actual `{k_min,
k_min+0.5, ..., k_max}` set from Eq. 30.

**Run:** `uv run python orbit/k_bounds.py`
**Writes:** `orbit/k_bounds.npz`, `k_bounds.txt`
**Reads:** `observation/generated_attributables.npz`; imports `A_MIN`,
`A_MAX` from `range_bounds.py`

### 5. `dynamic/lambert.py` -- boundary-value orbit solver

`lambert_boundary_value(mu, r1, r2, tof, k)` is Section 5's "Orbit" step
(Fig. 5): given two hypothesized position vectors, a flight time, and a `k`,
solves Lambert's problem for the connecting two-body orbit(s). Wraps
`hapsira`'s Izzo-algorithm implementation, but does *not* trust its
`prograde`/`low_path` flags to mean "short way"/"long way" directly (that
turned out to be geometry-dependent, not a fixed convention -- see the
module's own docstring for the full explanation). Instead it tries all
feasible flag combinations and classifies each *result* by directly
recomputing its actual swept transfer angle, independent of `hapsira`'s
internal sign conventions.

Returns a **list** of `(v1, v2)` candidates, not a single pair: for `M = 0`
there's always exactly one, but for `M > 0` there can genuinely be two
different real orbits (the "vacant focus" ambiguity) satisfying the same
`k`. The paper doesn't disambiguate between them for its near-GEO case; this
is left for the discriminator stage.

**Run standalone:** `uv run python dynamic/lambert.py` runs its self-test (a
published textbook example, plus a round-trip check against this project's
own `two_body` propagator across short-way/long-way and 0/1-revolution
cases). It is otherwise a library module, imported by the next stage.

### 6. `hypothesis/build_hypothesis.py` -- ties it together

Implements Eq. 5, `r(rho) = r_site + rho * u`, then hands the result to
`lambert_boundary_value`. `build_hypothesis()` takes two tracklets' geometry
and a `(rho_1, rho_2, k)` guess, orders them by epoch internally (so callers
don't need to know which tracklet came first), and returns the candidate
orbit(s).

Its `main()` demonstrates the full chain on the real simulated data --
`range_bounds.npz` + `k_bounds.npz` in, candidate orbits out -- using the
*midpoint* of each tracklet's `[rho_min, rho_max]` as a placeholder guess.
That midpoint is explicitly **not** the paper's circular-orbit initial guess
(Eq. 31); that belongs to the optimizer stage, not yet built.

**Run:** `uv run python hypothesis/build_hypothesis.py`
**Reads:** `orbit/range_bounds.npz`, `orbit/k_bounds.npz`; imports from
`dynamic/lambert.py` and `orbit/k_bounds.py`

## Running the whole pipeline

From the project root, in dependency order:

```bash
uv run python simulation/new_data_gen.py
uv run python observation/create_attributes_vector.py
uv run python orbit/range_bounds.py
uv run python orbit/k_bounds.py
uv run python hypothesis/build_hypothesis.py
```

Every script resolves its file paths relative to its own location on disk
(`Path(__file__).resolve().parent`), not the shell's current directory, so
this works identically whether you run it from the project root (as above)
or `cd` into each stage's own directory first.

## What's not built yet

- **Discriminator loss (Eq. 16-19).** From a Lambert candidate's `v1, v2`,
  compute the *modeled* angular rates at each site, propagate the
  attributable covariance through to get the modeled-rate covariance, and
  form the weighted loss `L(p, k)` comparing modeled vs. *measured* angular
  rates -- this is the actual accept/reject signal.
- **Optimizer (Eq. 30-31).** For each `k`, minimize `L(p, k)` over
  `p = (rho_1, rho_2)` with a quasi-Newton method (BFGS), seeded from the
  circular-orbit initial guess of Eq. 31.
- **Gating / decision.** Threshold `min_k L` to accept or reject a tracklet
  pair; handle multiple `k` candidates passing the threshold.
- **Multi-object simulation.** `new_data_gen.py` currently generates a
  single object, so every pair `k_bounds.py` produces today is a true
  association. Reproducing anything like the paper's Fig. 9 (false
  positive/negative rates) needs multiple objects, some genuinely unrelated.

## Notes and gotchas

- **`hapsira` is an added dependency**, chosen over hand-rolling the Lambert
  solver after checking it against the project's existing minimal-dependency
  philosophy: it's MIT-licensed, and only its low-level
  `hapsira.core.iod.izzo` (Numba-jitted, plain-array) entry point is used,
  not the heavier astropy-Quantity-wrapped public API, since this will be
  called many times inside the eventual optimizer loop.
- **Python was bumped from 3.9 to 3.11.** The originally pinned
  `astropy>=5.3.2` predates numpy 2.0 support (which requires
  `astropy>=6.1`, which in turn requires Python >=3.10). This was a
  pre-existing break discovered while verifying the Lambert solver, not
  something introduced by it.
- **`new_data_gen.py` overwrites, not appends**, despite what `README.md`
  (which documents the now-deleted `data_generation (1).py`) says. Its
  random seed is fixed, so this is actually the more useful behavior for
  this pipeline: reruns are deterministic and idempotent rather than
  accumulating stale rows.
- **`simulation/data_generation.py`** is the direct predecessor of
  `new_data_gen.py` and is no longer used by anything; kept around only as
  history.
