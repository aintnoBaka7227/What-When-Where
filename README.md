# What When Where

Short-arc tracklet association for geostationary objects — an implementation of
the boundary-value method from:

> J.A. Siminski, O. Montenbruck, H. Fiedler, T. Schildknecht, "Short-arc
> tracklet association for geostationary objects," *Advances in Space Research*
> 53 (2014) 1184–1194. (`sources/siminski.pdf`)

**The problem.** An optical telescope sees a satellite for a minute or two — a
*tracklet* of 3–5 right-ascension/declination measurements — then loses it.
Later, something is seen again. Are the two tracklets the same object? Angles
alone constrain only 4 of an orbit's 6 degrees of freedom, so the question is
underdetermined.

**The approach.** Hypothesise the two unknown ranges, solve Lambert's problem to
get a candidate orbit, then check whether that orbit's *predicted* angular rates
match the *measured* ones. Optimise over the hypothesised ranges and the
revolution count `k`; a small enough mismatch means the tracklets associate.

For the full stage-by-stage description with every input and output, see
[PIPELINE.md](PIPELINE.md).

## Project structure

Code lives in the `lab` package under `src/lab/`, with one directory per pipeline
phase. Stage numbers are the order the stages run in. Generated files go in
`data/`, and tests go in `test/`.

| Directory | Stage | What lives there |
|---|---|---|
| `src/lab/data/` | 1 | Synthetic observation generation. `gcrs_data_generation.py` is the generator; `dataset.py` defines the chains and reads their files. Files prefixed `stale_` are superseded generators, kept for reference. |
| `src/lab/observation/` | 2 | Turning raw RA/Dec series into *attributables*. `fit_attributes.py` is the fit itself; `create_attributes_vector.py` drives it over the dataset. |
| `src/lab/hypothesis/` | 3–4 | The admissible search region. `range_bounds.py` bounds the topocentric range; `k_bounds.py` bounds the revolution count. |
| `src/lab/model/` | 5–6 | Candidate orbits and their predictions. `lambert.py` is the boundary-value solver; `build_orbit.py` wires it to the bounds; `model_rates.py` gives the angular rates an orbit *predicts*. |
| `src/lab/scoring/` | 7 | `discriminator_loss.py` — how badly predicted rates disagree with measured ones. |
| `src/lab/search/` | 8 | `optimise_hypothesis.py` — searching the region for the best-fitting orbit. The actual association test. |
| `src/lab/evaluation/` | — | `validate_optimiser.py`, the optimiser diagnostic. |
| `src/lab/runner.py` | — | Run folders: hashing, config, logs, parent links. |
| `src/lab/paths.py` | — | `PROJECT_ROOT`, `DATA_DIR`, `RESULTS_DIR`, `RUNS_DIR`, `FIGURES_DIR`. |
| `experiment/run/` | — | `run_<stage>.py`: runs one stage and saves its result. |
| `experiment/prod/` | — | `prod_<stage>.py`: draws figures from a stage's saved result. |
| `results/` | — | Run folders and figures, keyed by run hash (not tracked). |
| `data/` | — | Generated data outside runs: the GCRS dataset, and the output of stages run by hand. |
| `test/` | — | Test suites. |
| `sources/` | — | The paper. |

Two files are worth knowing about before you read any others:

- **`src/lab/hypothesis/range_bounds.py`** is the path authority for the whole project.
  It defines `VARIANTS` and the `*_FILES` dictionaries that stages 3–8 import
  rather than building paths themselves. If you want to know which files a stage
  reads, start there.
- **`src/lab/model/lambert.py`** is self-contained and has no pipeline dependencies. It
  is the one piece you can read and test in isolation.

## The chains

Every stage runs **once per chain**, over parallel datasets generated from the
same orbits, stations and epochs:

- **clean** — exact angles, no measurement error.
- **noisy_0.7as, noisy_1.2as, noisy_2.0as** — the same angles with Siminski
  measurement error applied, at the paper's three relative noise levels, each
  with 5″ systematic error.

They differ only in the two measured-angle columns, so any result can be
compared against its noise-free counterpart and across noise levels. The chains
are defined in `src/lab/data/dataset.py`, and files are suffixed accordingly.

The clean chain is for geometry debugging — with no noise, any residual is a
bug in the code, not in the data. The noisy chains are the real experiment.

## Running it

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

This also installs the `lab` package in editable mode, which is how the stages
import each other.

Then run a stage. Each stage has its own script, `experiment/run/run_<stage>.py`,
and there are two ways to use it:

```bash
# On its own: run this stage and everything upstream of it
uv run python experiment/run/run_search.py -n 4

# On a previous stage's output
uv run python experiment/run/run_search.py --parent <hypothesis hash>
```

With `-n`, upstream stages that already completed with the same settings are
reused, so `run_search.py -n 4` is also how you run the whole pipeline.

| Script | Stage | Input (`--parent`) | Result |
|---|---|---|---|
| `run_data.py` | [1] | none (`-n` only) | `generated_data_gcrs_*.txt` and a settings record |
| `run_observation.py` | [2] | data | `generated_attributables_*` |
| `run_hypothesis.py` | [3], [4] | observation | `range_bounds_*`, `k_bounds_*` |
| `run_model.py` | [5], [6] | hypothesis | `model_*.npz`: candidate orbits and predicted rates at the range-bound midpoints |
| `run_scoring.py` | [7] | hypothesis | `surface_*.npz`: loss surfaces |
| `run_search.py` | [8] | hypothesis | `association_*.npz`: best hypothesis per pair |
| `run_evaluation.py` | — | search | `optimiser_history_*.csv` |

Every run prints a 12-character hash and has its own folder:

```
results/runs/<hash>/
├── config.json     stage, parent, parameters, code hash, status, result summary
├── run.log         the terminal output
└── output/         this stage's files, plus links to its inputs
```

`config.json` lists the files the stage itself produced under `outputs`. The hash
covers the stage, its parameters, its parent and the code in `src/lab/` and
`experiment/run/`. A completed config is reused rather than repeated, unless you
pass `--force`, because generation is seeded and the optimiser is deterministic.
Any code change gives new hashes. `config.json` also records the Git commit.

Figures are drawn from a run's saved output, into `results/figures/<hash>/`:

```bash
uv run python experiment/prod/prod_search.py <search hash>      # Fig. 9
uv run python experiment/run/run_scoring.py --parent <hypothesis hash> --pair 0.0-0.2
uv run python experiment/prod/prod_scoring.py <scoring hash>    # Fig. 7
uv run python experiment/run/list_runs.py                       # show every run
```

Use at least 2 objects, or stage 8 has only same-object pairs and nothing to
discriminate against. Ten objects gives 50 tracklets and 1,225 pairs, which is a
reasonable working size and takes about 2 minutes.

Each stage can still be run by hand, in which case it reads and writes `data/`:

```bash
uv run python src/lab/data/gcrs_data_generation.py  # 1  appends ONE object
uv run python src/lab/observation/create_attributes_vector.py  # 2
uv run python src/lab/hypothesis/range_bounds.py    # 3
uv run python src/lab/hypothesis/k_bounds.py        # 4
uv run python src/lab/model/build_orbit.py          # 5
uv run python src/lab/search/optimise_hypothesis.py # 8  the association test
```

Stages 6 and 7 are libraries with nothing to run.

`test/test_lambert.py` is a self-test rather than a pipeline stage; run it
whenever you touch the solver.

## Conventions

- Positions in km, velocities in km/s, angles in radians, time in seconds.
- `mu = GM_earth / 1e9`, converting astropy's SI value to km³/s².
- Attributable ordering is fixed project-wide: `[alpha, alpha_dot, delta, delta_dot]`.
- `k` encodes both revolutions and transfer direction: integer part is the
  number of full revolutions, fractional part (0 or 0.5) selects short-way or
  long-way.
- Stage 1 writes `.txt` at full precision, which is canonical and reads back
  exactly. Later stages write a `.npz` for computation and a `.txt` for reading.

## Requirements

Python ≥ 3.11 (astropy's numpy-2.0 support requires it). Key dependencies:
`numpy`, `scipy`, `numba` (JIT for the two-body propagator), `astropy` (Earth
constants and frame transforms). Full list in `pyproject.toml`.
