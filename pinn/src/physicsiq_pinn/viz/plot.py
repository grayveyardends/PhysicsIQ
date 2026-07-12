"""plot.py — quick matplotlib pictures. Nothing here is load-bearing;
the real visualization is the heatmap overlay inside FreeCAD."""

import matplotlib

matplotlib.use("Agg")  # headless backend: we save files, never open windows
import matplotlib.pyplot as plt  # noqa: E402


def scatter_field(points, values, path, title="von Mises (MPa)"):
    """2D or 3D point cloud colored by stress. 3D is projected which is
    crude but enough for a sanity check."""
    fig, ax = plt.subplots(figsize=(7, 6))
    if points.shape[1] == 3:
        sc = ax.scatter(points[:, 0], points[:, 2], c=values, s=6, cmap="jet")
        ax.set_xlabel("x (mm)"), ax.set_ylabel("z (mm)")
    else:
        sc = ax.scatter(points[:, 0], points[:, 1], c=values, s=6, cmap="jet")
        ax.set_xlabel("x̂"), ax.set_ylabel("ŷ")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"plot saved: {path}", flush=True)


def loss_curves(history, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = [k for k in history[0] if k != "epoch"]
    for k in keys:
        ax.semilogy([h["epoch"] for h in history], [h[k] for h in history],
                    label=k)
    ax.set_xlabel("epoch"), ax.set_ylabel("loss (log)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
