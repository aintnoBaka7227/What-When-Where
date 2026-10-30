"""Fig. 7 from a scoring run's saved surfaces, into results/figures/<hash>/.

    uv run python experiment/prod/prod_scoring.py <scoring hash>
"""

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lab.paths import FIGURES_DIR, PROJECT_ROOT
from lab.runner import load_run


def figure_dir(run_id):
    path = FIGURES_DIR / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_surface(surface, out_path):
    """Draw the Fig. 7 panels for one saved pair."""
    grid_k = surface["k_grid"]
    r1, r2 = surface["rho_1"], surface["rho_2"]
    rho1_true, rho2_true = surface["truth"]
    variant, tag = str(surface["variant"]), str(surface["tag"])

    # One panel per k, side by side. Each gets its own colourbar because the
    # panel holding the minimum spans several orders of magnitude while the
    # others are near-flat plateaus -- a shared scale would wash it out.
    fig, axes = plt.subplots(1, len(grid_k), figsize=(4.6 * len(grid_k), 4.4),
                             squeeze=False)
    for index, (ax, k) in enumerate(zip(axes[0], grid_k)):
        # Ranges drawn in 10^4 km to match the paper's axes.
        im = ax.pcolormesh(r1 / 1e4, r2 / 1e4, np.log10(surface["loss"][index]),
                           shading="auto", cmap="Greys_r")
        fig.colorbar(im, ax=ax, label=r"$\log_{10} L(\mathbf{p})$")
        ax.plot(rho1_true / 1e4, rho2_true / 1e4, "o", ms=9, mfc="none",
                mec="#d1495b", mew=1.8, label="truth")
        if np.all(np.isfinite(surface["seeds"][index])):
            ax.plot(*(surface["seeds"][index] / 1e4), "P", ms=8,
                    color="#3a7ca5", label="Eq. 31 seed")
        if np.all(np.isfinite(surface["optima"][index])):
            ax.plot(*(surface["optima"][index] / 1e4), "x", ms=9, mew=2,
                    color="#2a9d3a", label="optimiser L*")
        ax.set_title(f"$k \\in [{k}, {k + 0.5}]$")
        ax.set_xlabel(r"$\rho_1$  [$10^4$ km]")
        ax.set_ylabel(r"$\rho_2$  [$10^4$ km]")
    axes[0][0].legend(loc="upper left", fontsize=8, framealpha=0.9)

    kind = "TRUE" if surface["is_true"] else "false"
    fig.suptitle(f"Fig. 7 analogue -- {tag} [{kind}], {variant} chain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure7(args):
    """Reproduce Siminski et al. Fig. 7 for this pipeline.

    Fig. 7 is the boundary-value loss L(p, k) for ONE tracklet pair, drawn as
    a heatmap over the range hypothesis p = (rho_1, rho_2), with one panel per
    k in the pair's admissible grid. The paper's point: unlike the
    initial-value method, each fixed-k slice has a single minimum, and the
    overall solution is the k whose minimum is lowest.

    Markers: the true (rho_1, rho_2) from the synthetic truth, the Eq. (31)
    circular-orbit seed, and the optimiser's result at that k. One figure is
    written for each surface in the run.
    """
    output_dir = load_run(args.hash, "scoring")
    out_dir = figure_dir(args.hash)

    for path in sorted(output_dir.glob("surface_*.npz")):
        with np.load(path) as surface:
            out = out_dir / f"figure7_{surface['variant']}_{surface['tag']}.png"
            plot_surface(surface, out)
        print(f"wrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw Fig. 7 from a scoring run.")
    parser.add_argument("hash", help="hash of a scoring run")
    figure7(parser.parse_args())
