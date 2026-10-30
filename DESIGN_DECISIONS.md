# Design Decisions

Decisions behind this project and the reasons for them. Equation, section and
table numbers refer to Siminski et al. (2014), `sources/siminski.pdf`.

## Pipeline phases

The pipeline is divided into eight phases, each with one job. Phase names describe
responsibility; stage numbers [1]–[8] are the execution order used in
[PIPELINE.md](PIPELINE.md) and [README.md](README.md). Stage 6 of `main.py`'s
run list is stage [8] here, because `main.py` leaves out the two library stages.

| Phase | Question it answers | Stage | Code | Output | Status |
| --- | --- | --- | --- | --- | --- |
| **Data** | What was observed, and what is the truth? | [1] | `src/lab/data/` | Observation rows, clean and noisy | Built |
| **Observation** | What does each tracklet measure? | [2] | `src/lab/observation/` | Attributable and covariance per tracklet | Built |
| **Hypothesis** | Which orbits are admissible? | [3], [4] | `src/lab/hypothesis/` | Range bounds per tracklet, `k` bounds per pair | Built |
| **Model** | What would a candidate orbit look like? | [5], [6] | `src/lab/model/` | Lambert orbits and predicted angular rates | Built |
| **Scoring** | How badly does a candidate disagree with the measurement? | [7] | `src/lab/scoring/` | Loss for one candidate | Built |
| **Search** | Which candidate fits a pair best? | [8] | `src/lab/search/` | Best `(k, rho_1, rho_2, L*)` per pair | Built |
| **Decision** | Are the two tracklets the same object? | None | None | Accept or reject per pair | **Not built** |
| **Evaluation** | How good are the results? | None | `run_association_test`, `src/lab/evaluation/` | True/false loss ranges, FP/FN rates, optimiser traces | Partly built |

### Project layout

| Decision | Reason |
| --- | --- |
| Code lives in `src/lab/<phase>/`, one directory per phase | The directory says what a module is responsible for. Decision has no directory until it has code. |
| `lab` is an installed package, and modules import as `lab.<phase>.<module>` | Replaces per-script `sys.path` edits. `uv sync` installs it in editable mode. |
| Stage scripts run by file path | `main.py` runs each one as a separate process, so a failure stays isolated. |
| Generated files go in `data/`, figures in `figures/`, logs in `runs/` | `src/` holds only code. `src/lab/paths.py` defines these locations. |
| Tests go in `test/` | One place for all suites. |
| `main.py` stays at the project root | It orchestrates the stages and imports none of them. |
| `range_bounds.py` remains the file registry | Moved unchanged in role. `create_attributes_vector.py` still keeps its own list, so stage 2 does not depend on stage 3. |

### Boundaries

- **Data** is the only phase that may use the true orbit. Columns 0–5 of the data
  matrix exist for scoring, so only Evaluation may read them.
- **Observation** is the only phase that fits measurements. Everything after it
  uses attributables, plus the station state from Data.
- **Hypothesis** defines the search space but evaluates no orbit. Its bounds must
  match the population drawn in Data.
- **Model** and **Scoring** are libraries called by Search. They write no files.
- **Search** returns a loss and does not decide. A threshold belongs in Decision.
- **Decision** must use a separate threshold per chain, because clean and noisy
  losses have different units.
- **Evaluation** consumes results and feeds nothing back into the method.

`run_association_test` currently sits inside Search but does Evaluation work: it
compares results against `object_id`. `src/lab/evaluation/figure9_fp_fn.py` sweeps thresholds
to plot FP/FN rates, but that is a measurement and does not make a decision.

### Clean and noisy chains

Every phase runs over both chains, which share everything except the measured
angles. Scoring is where their method differs: clean uses the unweighted loss and
noisy uses the weighted loss. Adding the 1.2″ and 2.0″ noise levels means adding
chains, which currently requires editing `VARIANTS` in `range_bounds.py` and
`DATASETS` in `create_attributes_vector.py`.

### Current state of the Data phase

`src/lab/data/new_data_gen.py` is still the generator the pipeline calls.
`src/lab/data/gcrs_data_generation.py` is intended for this phase but does not yet
produce what Observation reads. The pipeline's `generated_data_{clean,noisy}.npz`
files were removed in `a93b7e9`, so it needs `main.py --objects N` to run.

## Data generation: `src/lab/data/gcrs_data_generation.py`

Status as of commit `a93b7e9`. This generator is not connected to the pipeline
yet: stages 2 onward still read the `.npz` files written by
`src/lab/data/new_data_gen.py` (see [Open issues](#open-issues)).

### Purpose

Produce synthetic angles-only tracklets with known truth, so that association
results can be scored against the true object. Stations are Earth-fixed with a
physically correct GCRS position and velocity, because the station velocity is
the `r_s-dot` term in Eq. (6).

### Flow

Each run generates one object:

1. Draw classical orbital elements for the chosen test case.
2. Convert them to equinoctial elements, then to a Cartesian GCRS state.
3. For each of the 5 tracklets, draw a station and one systematic bias.
4. For each of the 5 observations, propagate the satellite, get the station's
   GCRS state, compute the RA/Dec, and append one row to every data file.
5. Add the object's drawn values to the settings record.

### Orbit

| Decision | Value | Reason |
| --- | --- | --- |
| Frame | GCRS axes | Astropy provides the station state in this frame directly. |
| Dynamics | Two-body, point-mass Earth | Matches the Lambert-based orbit model, so clean-chain residuals point to code bugs rather than unmodelled forces. |
| `mu` | astropy `GM_earth`, in km³/s² | Same source as `src/lab/model/build_orbit.py` and `src/lab/scoring/discriminator_loss.py`. |
| Semi-major axis | 38,000–45,000 km | The bounds the paper uses for its association tests (Section 6.2). |
| Eccentricity | 0–0.01 (test case `'a'`) | Near-circular. Cases `'b'`–`'d'` exceed `E_MAX = 0.01` in `src/lab/hypothesis/range_bounds.py`. |
| Inclination | 1–179° | Avoids the equatorial singularity in the classical elements. Wider than a near-GEO population; see [Open issues](#open-issues). |
| RAAN, argument of perigee, true anomaly | Uniform on 0–2π | No preferred geometry. |
| Epoch | 2012-08-18T22:59:08 UTC | First frame in Table 2. |

Semi-major axis is drawn through perigee radius, `rp ~ U(a_min(1-e), a_max(1-e))`,
which gives a uniform `a` over the bounds for the drawn `e`.

### Observation geometry

| Decision | Value | Reason |
| --- | --- | --- |
| Tracklets per object | 5 | Yields pairs at several separations. |
| Tracklet start spacing | 13 h | Pairs are 13, 26, 39 and 52 h apart. Multiples of 12 h are close to multiples of half a GEO period, where Lambert's problem is degenerate. |
| Observations per tracklet | 5 | Upper end of Table 2's 3–5. |
| Tracklet duration | 80 s, sampled every 20 s | Table 2's 1–2 min, and the ~20 s spacing in Section 6. |

At exactly GEO (`a = 42,164 km`), the 13 h spacing keeps transfer angles 15.5°,
31.1°, 46.6° and 62.1° away from 0°/180°. This protection is specific to periods
near 24 h; see [Open issues](#open-issues).

The total span of 52 h is longer than the ~27 h of the paper's data set (Table
2). The paper does discuss separations of up to two days (Section 6.2).

### Stations

| Decision | Value | Reason |
| --- | --- | --- |
| Stations | One new random station per tracklet | Guarantees every tracklet is observable without modelling a real site's visibility windows. The paper's data comes from a single telescope. |
| Placement | Sub-satellite direction rotated by up to 45° about a random axis | Keeps the object high in the sky. The lowest possible elevation is ~37°; the committed object ranges from 52.8° to 89.5°. |
| Radius | `R_earth` sphere, then attached to ITRS | The station is a geocentric point, so its height above the WGS84 ellipsoid varies (e.g. +7 km). This has no effect on the geometry the pipeline uses. |
| Earth attachment | GCRS → ITRS once at the first observation (`fix_station_to_earth`) | The station then stays on the same patch of ground for the whole tracklet. |
| Station state | `EarthLocation.get_gcrs_posvel` at every observation (`station_gcrs_state`) | Includes Earth rotation, precession, nutation and polar motion, and supplies the velocity for Eq. (6). |

### Measurement model

Angles are geometric: `RA/Dec = direction(r_sat - r_site)`. Light-time, aberration
and refraction are not modelled.

`measured = true + bias_tracklet + sigma_relative * unit_noise_observation`

| Decision | Value | Reason |
| --- | --- | --- |
| Relative noise | 0.7″, 1.2″, 2.0″ | The levels tested in Section 6.1. |
| Systematic noise | 5″ | Section 6.1. |
| Systematic draw | Once per tracklet, for RA and Dec separately | Section 6 describes an exposure-epoch offset, which stays constant across a tracklet. The model applies it as an angle bias rather than a timing error. |
| Shared unit draws | Noise is drawn once and scaled to each level | Files differ only in noise amplitude, so a change in results between levels comes from sigma rather than chance. |
| RA noise | Not divided by `cos(dec)` | Matches `new_data_gen.py` and the fitter. On-sky RA noise therefore shrinks at high declination. |
| RA wrap | `% 2π` after noise | Keeps RA in 0–2π. `src/lab/observation/fit_attributes.py` unwraps it before fitting. |

### Output files

All files are in `data/`.

| File | Contents |
| --- | --- |
| `generated_data_gcrs_clean.txt` | True angles |
| `generated_data_gcrs_noisy_{0.7,1.2,2.0}as.txt` | Noisy angles at one noise level |
| `generated_data_gcrs_settings.json` | Run-independent settings, plus each object's drawn orbit, stations and biases |

The files are row-aligned and differ only in columns 6–7:

| Columns | Contents | Units |
| --- | --- | --- |
| 0–2 | Satellite position (GCRS) | km |
| 3–5 | Satellite velocity (GCRS) | km/s |
| 6–7 | Topocentric RA, Dec | rad |
| 8–10 | Station position (GCRS) | km |
| 11–13 | Station velocity (GCRS) | km/s |
| 14 | Time since epoch | s |
| 15–17 | `tracklet_id`, `object_id`, `observation_id` | |

Values are written with full float precision, unlike the `%f` text copies from
`new_data_gen.py`, which quantise the angles.

### Appending and integrity

Each run appends one object, and IDs continue from the last row. Two guards stop
a dataset being silently corrupted:

- The script compares `next_ids` across all data files at startup and raises if
  they are out of step. `next_ids` itself raises on a wrong column count.
- `load_settings_record` raises if the settings differ from those recorded, or
  if the record's object count differs from the data. Because the record is
  written last, this also catches a run that crashed partway.

To reset, delete all `data/generated_data_gcrs_*` files.

### Open issues

**Not connected to the pipeline.** `create_attributes_vector.py` and
`range_bounds.py` read `data/generated_data_{clean,noisy}.npz`, with a
`data` array plus `relative_uncertainty_arcsec` and
`systematic_uncertainty_arcsec`. This generator writes text files to `data/`,
and the pipeline only has `clean` and `noisy` variants, not three noise levels.

**Generation is not reproducible.** It uses the unseeded global `np.random`,
while `new_data_gen.py` uses `default_rng(object_id)`. The settings record stores
the orbit, stations and biases, but not the per-observation noise draws.

**The 13 h spacing does not protect the whole semi-major axis range.** Near
`a = 44,556 km` the period is 26 h, so all four separations are multiples of half
a period. At `a = 38,398 km` the 52 h separation is too.

| Near-degenerate pair (within X of 0°/180°) | Share of the `a` range |
| --- | --- |
| Within 1° | 5.5% |
| Within 5° | 21.9% |
| Within 10° | 37.0% |

The committed object (`a = 43,774 km`) has a 175.6° transfer angle at 13 h. The
effect on the Lambert solver and on stage 8 has not been measured.

**Recorded station offset is not the true offset.** The rotation axis is not
perpendicular to the sub-satellite direction, so the true separation is at most
the rotation angle. `station_offset_from_subsatellite_deg` stores the rotation
angle: tracklet 0 records 36.39° where the true separation is 16.10°. Under this
sampling the true separation has a median of 16.7° and a 95th percentile of 37.9°.
The 45° limit still holds. `new_data_gen.py` samples the same way.

**Inclination is not near-GEO.** The 1–179° range includes polar and retrograde
orbits, which are easier to separate than the paper's population.

**Objects are independent draws**, so false pairs are easier than the paper's
hard case of similar orbits.

**Unused code carries a bug.** `gcrs_to_oe` computes `s2 = 1 + h + k`; the
standard form is `1 + h² + k²`. Nothing calls it. In `two_body`, the
`if (i == imax)` warning can never fire, because `i` stops at `imax - 1`.
