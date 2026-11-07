# Design Decisions

Decisions behind this project and the reasons for them. Equation, section and
table numbers refer to Siminski et al. (2014), `sources/siminski.pdf`.

## Pipeline phases

The pipeline is divided into eight phases, each with one job. Phase names describe
responsibility; stage numbers [1]–[8] are the execution order used in
[PIPELINE.md](PIPELINE.md) and [README.md](README.md).

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
| Stage scripts run by file path | `src/lab/runner.py` runs each one as a separate process, so a failure stays isolated. |
| Run outputs go in `results/`, and stages run by hand write to the untracked `data/` | `src/` holds only code. `src/lab/paths.py` defines these locations. |
| Tests go in `test/` | One place for all suites. Every suite generates into a temporary folder, so it needs no tracked data. |
| Stage paths are `lab.paths.VariantFiles`, resolved when indexed | Repointing `DATA_DIR` after import then moves every stage. Fixed at import, they could silently read `data/` instead of a run's folder. |
| Data columns are named in `lab.data.dataset` | One layout, checked against the generator, instead of bare indices in each stage. |
| No generated data is tracked in Git | Every dataset is reproducible from its config, and runs keep their own. |
| `range_bounds.py` remains the file registry | Moved unchanged in role. `create_attributes_vector.py` still keeps its own list, so stage 2 does not depend on stage 3. |

### Experiments and results

| Decision | Reason |
| --- | --- |
| One `experiment/run/run_<stage>.py` per stage | Each stage can be run, and its result inspected, by itself. Together they replace `main.py`. |
| One `experiment/prod/prod_<stage>.py` per stage with figures, reading only a run's saved files | A figure always shows a specific recorded result, and redrawing it repeats no computation. |
| A stage takes `--parent <hash>`, or `-n N` to build everything upstream | A stage can reuse a chosen result or run from nothing. Both modes share one code path. |
| Parents follow what a stage reads, not stage number | Model, Scoring and Search each read Hypothesis's output. They use each other as libraries only. Evaluation reads Search. |
| Stages 5–7 save results through their run scripts | `run_model.py` saves candidates at the range-bound midpoints, and `run_scoring.py` saves loss surfaces. Those stages otherwise persist nothing. |
| Each run has a folder `results/runs/<hash>/` with `config.json`, `run.log` and `output/` | A result stays with what produced it, and runs cannot overwrite each other. |
| `output/` links to the parent's files, and `config.json` lists the stage's own in `outputs` | Stage code reads and writes one folder, so it runs unchanged. Deleting a parent run breaks its children. |
| The hash covers stage, parameters, parent and a content hash of `src/lab/` and `experiment/run/` | Uncommitted edits are captured too, which a Git commit alone would miss. `config.json` records the commit for reference. |
| A completed config is reused without `--force` | Generation is seeded and the optimiser is deterministic, so the result would be identical. |
| Stages find their folder through `LAB_DATA_DIR` | Stage scripts stay runnable by hand, against `data/`. |
| Figures go in `results/figures/<hash>/` | The same hash links a figure to its run. |
| Output from before this scheme is in `results/{runs,figures}/legacy/` | It has no config, so it cannot be given a hash. |

### Metrics

| Decision | Reason |
| --- | --- |
| Each stage is judged on whether it preserved the truth | Simulation gives the true answer at every stage, so a problem can be traced to the stage that caused it. |
| Metrics live in `src/lab/evaluation/metrics.py`, and are saved as `metrics.json` in the run folder | Not in `output/`, whose files are linked into child runs. |
| Stage 8 saves a per-`k` trace | Its final answer alone cannot show whether a search actually ran. |
| Only metrics read the truth columns | The method itself stays blind to them. |

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
compares results against `object_id`. `experiment/prod/prod_search.py` sweeps thresholds
to plot FP/FN rates, but that is a measurement and does not make a decision.

### Clean and noisy chains

Every phase runs over all four chains, which share everything except the measured
angles. Scoring is where their method differs: clean uses the unweighted loss and
noisy uses the weighted loss. Adding a noise level means adding it to
`RELATIVE_NOISE_CASES_ARCSEC` in `lab.data.dataset`; every stage's paths are
built from `VARIANTS` there.

### Current state of the Data phase

`experiment/run/run_data.py` calls `src/lab/data/gcrs_data_generation.py`, and
later stages read its text files through `src/lab/data/dataset.py`. The earlier
generators are kept as `stale_new_data_gen.py` and `stale_data_generation.py`.

## Data generation: `src/lab/data/gcrs_data_generation.py`

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
| Inclination | 0–20° | A near-GEO population, as in the paper. Orbits this similar make false pairs harder to reject. The elements are converted through equinoctial form, which has no singularity at 0°. |
| RAAN, argument of perigee, true anomaly | Uniform on 0–2π | No preferred geometry. |
| Epoch | 2012-08-18T22:59:08 UTC | First frame in Table 2. |

Semi-major axis is drawn through perigee radius, `rp ~ U(a_min(1-e), a_max(1-e))`,
which gives a uniform `a` over the bounds for the drawn `e`.

### Observation geometry

| Decision | Value | Reason |
| --- | --- | --- |
| Tracklets per object | 5 | Yields pairs at several separations. |
| Tracklet schedule | 0, 13, 29, 37, 53 h | Unequal gaps go degenerate at different orbital periods, so one period cannot ruin every pair of an object. |
| Jitter | Up to ±2 h on each tracklet after the first | Adds randomness. Simulated over the `a` range, it stays near the fixed schedule: 5.6% of pairs within 5° of 0°/180°, against 13.2% for even 13 h spacing, and 0.05% of objects lose half their pairs, against 11.8%. |
| Start offset | Up to 24 h per object | Objects are not observed at the same epochs, so no pair is dropped for having no time gap. |
| Observations per tracklet | 5 | Upper end of Table 2's 3–5. |
| Tracklet duration | 80 s, sampled every 20 s | Table 2's 1–2 min, and the ~20 s spacing in Section 6. |

The total span of about 53 h, plus the offset, is longer than the ~27 h of the paper's data set (Table
2). The paper does discuss separations of up to two days (Section 6.2).

### Stations

| Decision | Value | Reason |
| --- | --- | --- |
| Stations | One new random station per tracklet | Guarantees every tracklet is observable without modelling a real site's visibility windows. The paper's data comes from a single telescope. |
| Placement | Sub-satellite direction rotated by up to 45° about a random perpendicular axis | Keeps the object high in the sky, and makes the rotation angle the true offset. The lowest possible elevation is ~37°. |
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
| RA noise | Not divided by `cos(dec)` | Matches the fitter. With declination below about 20°, on-sky RA noise is at most about 6% smaller than intended. |
| RA wrap | `% 2π` after noise | Keeps RA in 0–2π. `src/lab/observation/fit_attributes.py` unwraps it before fitting. |

### Output files

All files are in the run's `output/` folder, or `data/` if the script is run by hand.

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

| Decision | Reason |
| --- | --- |
| Text files are canonical | Values are written at full precision and read back exactly, and they can be read directly. |
| One settings record per dataset | Holds the noise and population values that the rows cannot. `src/lab/data/dataset.py` supplies them to later stages. |
| One chain per noise level | All levels run together, so results can be compared across levels from one run. |
| Seeded with `default_rng(object_id)` | Object *n* is always identical, so a run hash identifies its data. |
| Noise draws are unit normals shared across levels | Levels differ only in amplitude. |

### Appending and integrity

Each run appends one object, and IDs continue from the last row. Two guards stop
a dataset being silently corrupted:

- The script compares `next_ids` across all data files at startup and raises if
  they are out of step. `next_ids` itself raises on a wrong column count.
- `load_settings_record` raises if the settings differ from those recorded, or
  if the record's object count differs from the data. Because the record is
  written last, this also catches a run that crashed partway.

Run scripts start each dataset empty. To reset a dataset made by hand, delete
`data/generated_data_gcrs_*`.

### Validation

| Layer | Where | What it checks |
| --- | --- | --- |
| Self-consistency | `test/test_gcrs_data.py` | The code does what it records. |
| Independent routes | `test/test_data_validation.py` | Numerical integration for orbits, ERFA called directly for Earth rotation, and astropy for angles. |
| Statistics | `metrics.json` from `run_data.py` | Each drawn quantity fits its distribution. Judge these on 100 or more objects. |
| Realism | `metrics.json` from `run_data.py` | Rates and arc lengths are plausible, and settings match the paper. |
| End to end | `metrics.json` from `run_scoring.py` | The loss scores the truth as a match. |

Light-time delay and aberration are not simulated. Their effects are around 2″ and
20″, but the pipeline ignores them too, so the two stay consistent. They would
matter with real telescope data.

### Open issues

**Degenerate pairs cannot be removed completely.** Every time gap is degenerate
for some orbital period. Randomised spacing limits how many pairs one object can
lose, and a flag on near-0°/180° pairs belongs in the Search stage.

**Objects are independent draws**, so false pairs are easier than the paper's
hard case of similar orbits.

**Unused code carries a bug.** `gcrs_to_oe` computes `s2 = 1 + h + k`; the
standard form is `1 + h² + k²`. Nothing calls it. In `two_body`, the
`if (i == imax)` warning can never fire, because `i` stops at `imax - 1`.
