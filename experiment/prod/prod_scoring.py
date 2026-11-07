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

    # One panel per k, each with its own colourbar: a shared scale would flatten
    # the panel holding the minimum.
    fig, axes = plt.subplots(1, len(grid_k), figsize=(4.6 * len(grid_k), 4.4),
                             squeeze=False)
    for index, (ax, k) in enumerate(zip(axes[0], grid_k)):
        # 10^4 km, as in the paper.
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
    """Siminski et al. Fig. 7: log L over (rho_1, rho_2), one panel per k.

    Marks the truth, the Eq. (31) seed and the search's result. One figure
    per saved surface.
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
