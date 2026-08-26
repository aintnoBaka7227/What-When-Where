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

The directory layout follows the data flow — each directory is one step in the
chain, and the stage number is the order you run them in.

| Directory | Stage | What lives there |
|---|---|---|
| `simulation/` | 1 | Synthetic observation generation. `new_data_gen.py` is the current generator; `data_generation.py` is the superseded original. |
| `observation/` | 2 | Turning raw RA/Dec series into *attributables*. `fit_attributes.py` is the fit itself; `create_attributes_vector.py` drives it over the dataset. |
| `hypothesis/` | 3–4 | The admissible search region. `range_bounds.py` bounds the topocentric range; `k_bounds.py` bounds the revolution count. |
| `orbit/` | 5 | Turning a range hypothesis into an actual orbit. `lambert.py` is the boundary-value solver; `build_orbit.py` wires it to the bounds. |
| `modelled/` | 6 | `model_rates.py` — what angular rates a candidate orbit *predicts*. |
| `discriminator/` | 7 | `discriminator_loss.py` — how badly predicted rates disagree with measured ones. |
| `optimisation/` | 8 | `optimise_hypothesis.py` — searching the region for the best-fitting orbit. The actual association test. |
| `sources/` | — | The paper. |

Two files are worth knowing about before you read any others:

- **`hypothesis/range_bounds.py`** is the path authority for the whole project.
  It defines `VARIANTS` and the `*_FILES` dictionaries that stages 3–8 import
  rather than building paths themselves. If you want to know where data lives,
  start there.
- **`orbit/lambert.py`** is self-contained and has no pipeline dependencies. It
  is the one piece you can read and test in isolation.

## The two chains

Every stage runs **twice**, over two parallel datasets generated from one
orbit, one set of stations and one set of epochs:

- **clean** — exact angles, no measurement error.
- **noisy** — the same angles with Siminski measurement error applied
  (0.7″ relative, 5″ systematic).

They differ only in the two measured-angle columns, so any result can be
compared against its own noise-free counterpart. Files are suffixed
accordingly: `generated_data_clean.npz`, `generated_data_noisy.npz`, and so on.

The clean chain is for geometry debugging — with no noise, any residual is a
bug in the code, not in the data. The noisy chain is the real experiment.

## Running it

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Then run the stages in order:

```bash
uv run python simulation/new_data_gen.py            # 1  appends ONE object
uv run python observation/create_attributes_vector.py  # 2
uv run python hypothesis/range_bounds.py            # 3
uv run python hypothesis/k_bounds.py                # 4
uv run python orbit/build_orbit.py             # 5
uv run python modelled/model_rates.py               # 6
uv run python discriminator/discriminator_loss.py   # 7
uv run python optimisation/optimise_hypothesis.py   # 8  the association test
```

Every stage handles both chains itself — there is no flag to set.

**Stage 1 appends one object per run.** Run it several times before stage 2, or
stage 8 has only same-object pairs and nothing to discriminate against. Four
objects gives 12 tracklets and 48 pairs, which is a reasonable working size.

To reset to an empty dataset, delete the generated files and start over:

```bash
rm -f simulation/generated_data_{clean,noisy}.{npz,txt}
```

Generation is fully deterministic — object *n* is seeded with `n`, so the same
object count always reproduces the same dataset.

`orbit/lambert.py` is a self-test rather than a pipeline stage; run it directly
whenever you touch the solver.

## Conventions

- Positions in km, velocities in km/s, angles in radians, time in seconds.
- `mu = GM_earth / 1e9`, converting astropy's SI value to km³/s².
- Attributable ordering is fixed project-wide: `[alpha, alpha_dot, delta, delta_dot]`.
- `k` encodes both revolutions and transfer direction: integer part is the
  number of full revolutions, fractional part (0 or 0.5) selects short-way or
  long-way.
- Every stage writes a `.npz` for computation and a `.txt` for reading. **The
  `.npz` is canonical**; the text copies are lossy and nothing reads them back.

## Requirements

Python ≥ 3.11 (astropy's numpy-2.0 support requires it). Key dependencies:
`numpy`, `scipy`, `numba` (JIT for the two-body propagator), `astropy` (Earth
constants and frame transforms). Full list in `pyproject.toml`.
