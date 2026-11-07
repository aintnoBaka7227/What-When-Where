# Agent Notes

Context for an agent picking this project up cold. Read
[README.md](README.md) for structure and [PIPELINE.md](PIPELINE.md) for
per-stage inputs and outputs — this file covers what those two don't: the
decisions already made, and the ways it is easy to break things.

## What this is

A from-scratch implementation of Siminski et al. (2014), "Short-arc tracklet
association for geostationary objects" (`sources/siminski.pdf`). Given two short
arcs of angles-only observations, decide whether they are the same object.

It is a research reimplementation, not a product. The paper is the
specification, and equation numbers in docstrings are load-bearing — they refer
to the actual paper and were checked against it. When something in the code
looks wrong, check the paper before assuming a bug: at least one deviation is
deliberate and documented (see "Eq. 18" below).

## Orientation, fastest path

1. `README.md` — directory-to-stage map.
2. `src/lab/hypothesis/range_bounds.py` top of file — the path registry every other
   stage imports. Tells you where all data lives.
3. `PIPELINE.md` — the stage you actually care about.

`src/lab/model/lambert.py` is the only module with no pipeline dependencies, and
`test/test_lambert.py` validates it. It is the safest place to start reading.

## The chain design

Everything runs once per chain: `clean`, `noisy_0.7as`, `noisy_1.2as` and
`noisy_2.0as`, generated together from the same orbits, stations and epochs. They
differ **only** in the two measured-angle columns. `src/lab/data/dataset.py`
defines them.

This is not redundancy. The clean chain is the debugging instrument: with no
measurement error, any nonzero residual is a code bug, not noise. It is how you
tell "the Lambert solver is wrong" from "the noise model is doing its job". The
noisy chains are the experiment.

Adding a stage means handling every variant. The established pattern is a
`build_x(variant)` or `demo_variant(variant)` function plus a `main()` that
loops over `VARIANTS` and skips any variant whose inputs are missing.

## Decisions already made — do not silently reverse

**Two loss functions, by necessity.** The clean dataset has zero measurement
uncertainty, so `fit_attributes` produces a zero covariance matrix, so the
weighted loss would invert a singular matrix. `discriminator_loss_unweighted`
exists for that reason. This was verified — the weighted loss on clean data
fails every pair with `LinAlgError: Singular matrix`. Do not "fix" this by
substituting a nominal sigma into the clean dataset without saying so; that
changes what the clean chain means.

**The unweighted loss is scored in arcsec²/s², not rad²/s².** This looks like an
arbitrary unit choice and is not. Near-GEO rates are ~1e−5 rad/s, so squared
residuals land near 1e−10, below L-BFGS-B's default `ftol` of ~2.2e−9. The
optimiser then declares convergence on its first step and returns the initial
guess. This was observed: `rho*` came back pinned at `rho_max`, ~3000 km from
truth, until the rescaling. If you change the units, re-verify that stage 8
recovers the true ranges.

**Losses are not comparable across chains.** Clean is arcsec²/s², noisy is
dimensionless. Any threshold must be chosen per chain.

**Eq. (18) includes a matrix inverse that the paper omits as printed.** This is
deliberate and argued in the `discriminator_loss.py` module docstring: the
paper's own Eq. (21)/(13) derivation and its chi-squared claim both require the
inverse. Do not "correct" the code to match the printed equation.

**`k_bounds` writes identical files for every chain.** `k` depends only on `t0`,
which the chains share. The duplication is intentional so each chain is
self-contained.

**`create_attributes_vector.py` keeps its own copy of the dataset paths** rather
than importing the registry from `range_bounds.py`. This was considered and kept
deliberately: importing would make stage 2 depend on stage 3, inverting the data
flow. Its list is built from `VARIANTS` in `lab.data.dataset`, which is stage 1,
so adding a variant means editing that one file.

## Verified behaviour, so you don't re-derive it

These were checked by running the code, not by reading it:

- **Generation is deterministic.** `rng = default_rng(object_id)`, so object *n* is
  always identical, and regenerating *n* objects reproduces the dataset byte for
  byte.
- **The propagator is sound.** Over 24 h, `two_body` conserves `a`, `e` and `h`
  to ~1e-10 relative.
- **Stations are real rotating-Earth sites.** `|v| = ω·r_eq` to 0.24%,
  `v · r = 3e-13`, radius ~6375 km.
- **Every object is observable.** Elevation stays above about 37° by
  construction — the station is placed within 45° of the sub-satellite point.
- **Both chains fully separate true from false pairs** on a four-object dataset.

## Working practices

Run stages with `uv run python <path>`; `src/lab/paths.py` resolves locations
from `__file__`, so the working directory doesn't matter. Stages import each
other as `lab.<phase>.<module>`, which needs the editable install from `uv sync`.

**Run stages through `experiment/run/run_<stage>.py`.** Each takes `--parent
<hash>` to use a previous stage's run, or `-n N` to run itself and everything
upstream. Each run gets `results/runs/<hash>/`, holding `config.json`, `run.log`
and `output/`. `output/` links to the parent's files, because a stage reads and
writes one folder; `config.json`'s `outputs` lists what the stage produced. The
hash covers the stage, parameters, parent and code, so any change under `src/lab/`
or `experiment/run/` gives new hashes. New figures go in
`experiment/prod/prod_<stage>.py` and must read a run's saved output rather than
recompute it.

**Index data rows by name.** `lab.data.dataset` defines the column layout
(`RA_COLUMN`, `SITE_POSITION`, `TIME_COLUMN`, …) and `tracklet_rows()`. Use those
rather than `data[:, 16]`; `test/test_dataset.py` checks them against the
generator.

**Stage paths resolve at call time.** The `*_FILES` mappings are
`lab.paths.VariantFiles` and read `lab.paths.DATA_DIR` when indexed. Tests use this:
`test/pipeline_fixture.py` generates a real dataset into a temporary folder and
runs stages 2–4 there in-process. See [TESTING.md](TESTING.md).

**Check `metrics.json` first.** `experiment/run/show_metrics.py <hash>` prints a
run's metrics and its ancestors'. Scoring is a sibling of Search, so show it
separately. Loss at the truth tells a data problem from a Search problem.

**Stage 1 appends one object per invocation.** `run_data.py -n N` handles
this. A single object gives only same-object pairs, so stage 8 reports
`False pairs: none` and tests nothing. Generate at least 3–4 objects before
drawing conclusions.

Stage 8 cost grows quadratically with tracklet count — four objects is ~90 s, so
background it rather than blocking.

**Stage 1's `.txt` files are canonical.** They are written at full precision and
read back exactly, and `generated_data_gcrs_settings.json` holds what rows cannot.
From stage 2 on, the `.npz` files are canonical and the `.txt` copies are for
eyeballing only, because they use `%f`.

**The orbit family is set in one place.** `gcrs_data_generation.py`'s
`test_case` is recorded in the settings file, and `population_for()` reads the
bounds from there.

## Known rough edges

- **No accept/reject gate.** Stage 8 produces a loss per pair; nothing turns it
  into a decision. This is the obvious next piece of work.
- **Objects are independent random draws**, not deliberately similar orbits, so
  the false-pair test is easier than the paper's hard case.
- **RA uncertainty is not divided by cos(δ)** in either the generator or the
  fitter. They are self-consistent, so nothing is wrong, but it diverges from
  standard astrometric convention if comparing to the paper's numbers directly.
- **Files prefixed `stale_` in `src/lab/data/` are dead.** Superseded generators,
  kept for reference. Not pipeline stages.
- **Stage 8 misses some true pairs.** In run `1e5c9c0a46f5`, 7 of 100 true pairs
  failed. All score near zero at the true ranges and `k`, so the data is correct
  and the failure is in Search. Most of the earlier 15 came from the Eq. (31) seed
  sitting at the lower edge of its revolution bin, a near-0° transfer. It now uses
  the bin midpoint. The rest have two causes, both measured in `test/test_search.py`:
  a flat penalty plateau with zero gradient, and a second basin in some
  multi-revolution landscapes. A sloped penalty or several seeds per `k` would be
  the next things to try.

## Note on this file

Claude Code auto-loads `CLAUDE.md`, not `AGENTS.md`. If you want this picked up
automatically every session, symlink it:

```bash
ln -s AGENTS.md CLAUDE.md
```
