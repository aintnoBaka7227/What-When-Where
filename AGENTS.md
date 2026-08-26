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
2. `hypothesis/range_bounds.py` top of file — the path registry every other
   stage imports. Tells you where all data lives.
3. `PIPELINE.md` — the stage you actually care about.

`orbit/lambert.py` is the only module with no pipeline dependencies, and it
self-tests. It is the safest place to start reading.

## The two-chain design

Everything runs twice, over a `clean` and a `noisy` dataset generated together
from the same orbit, stations and epochs. They differ **only** in the two
measured-angle columns.

This is not redundancy. The clean chain is the debugging instrument: with no
measurement error, any nonzero residual is a code bug, not noise. It is how you
tell "the Lambert solver is wrong" from "the noise model is doing its job". The
noisy chain is the experiment.

Adding a stage means handling both variants. The established pattern is a
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

**`k_bounds` writes identical clean and noisy files.** `k` depends only on `t0`,
which both chains share. The duplication is intentional so each chain is
self-contained.

**`create_attributes_vector.py` keeps its own copy of the dataset paths** rather
than importing the registry from `range_bounds.py`. This was considered and kept
deliberately: importing would make stage 2 depend on stage 3, inverting the data
flow. The cost is that adding a variant means editing two files.

## Verified behaviour, so you don't re-derive it

These were checked by running the code, not by reading it:

- **Generation is deterministic.** `rng = default_rng(object_id)`, so object *n*
  is always identical. Deleting everything and regenerating *n* objects
  reproduces the previous dataset byte for byte.
- **The propagator is sound.** Over 24 h, `two_body` conserves `a`, `e` and `h`
  to ~1e-10 relative.
- **Stations are real rotating-Earth sites.** `|v| = ω·r_eq` to 0.24%,
  `v · r = 3e-13`, radius ~6375 km.
- **Every object is observable.** Elevation stays between 49° and 89° by
  construction — the station is placed within 45° of the sub-satellite point.
- **Both chains fully separate true from false pairs** on a four-object dataset.

## Working practices

Run stages with `uv run python <path>`; they resolve their own paths from
`__file__`, so the working directory doesn't matter.

**Stage 1 appends one object per run.** A single object gives only same-object
pairs, so stage 8 reports `False pairs: none` and tests nothing. Generate at
least 3–4 objects before drawing conclusions. Reset with:

```bash
rm -f simulation/generated_data_{clean,noisy}.{npz,txt}
```

Stage 8 cost grows quadratically with tracklet count — four objects is ~90 s, so
background it rather than blocking.

**The `.npz` files are canonical.** The `.txt` copies use `%f` for angles, which
quantises RA/Dec at 0.206″ against a 0.7″ noise model. They are for eyeballing
only; nothing reads them back.

**Changing the orbit family means changing two places.** `new_data_gen.py`'s
`test_case` and `range_bounds.py`'s `A_MIN`/`A_MAX`/`E_MAX` must agree. Only
test case `'a'` matches the current bounds; `'b'`–`'d'` push eccentricity past
`E_MAX = 0.01`.

## Known rough edges

- **No accept/reject gate.** Stage 8 produces a loss per pair; nothing turns it
  into a decision. This is the obvious next piece of work.
- **Objects are independent random draws**, not deliberately similar orbits, so
  the false-pair test is easier than the paper's hard case.
- **Only the 0.7″ noise case exists.** The paper tests 0.7″, 1.2″ and 2.0″.
- **RA uncertainty is not divided by cos(δ)** in either the generator or the
  fitter. They are self-consistent, so nothing is wrong, but it diverges from
  standard astrometric convention if comparing to the paper's numbers directly.
- **`simulation/data_generation.py` is dead.** The superseded original
  generator, kept for reference. Not a pipeline stage.
- **One true pair sometimes converges on the wrong `k`.** The 24 h-separated
  pair has landed on `k=0.5` with a loss orders of magnitude worse than its
  siblings. Possibly a local minimum or a `k`-grid issue; uninvestigated.

## Note on this file

Claude Code auto-loads `CLAUDE.md`, not `AGENTS.md`. If you want this picked up
automatically every session, symlink it:

```bash
ln -s AGENTS.md CLAUDE.md
```
