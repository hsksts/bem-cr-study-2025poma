#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1-2_layout_plot_fibonacci.py

Publication-ready 3D visualization of:
- boundary mesh
- receiver grid
- Fibonacci-spiral source distribution

This script is intended for figure generation only.
No acoustic computation is performed here.

Pipeline position:
    0_generate_mesh.py
 -> 1_bem_export_space_fibonacci.py
 -> 1-2_layout_plot_fibonacci.py   (this script)
 -> 2_estimate_cr_from_space.py

Notes:
- Geometry path alternatives are intentionally left commented
  to indicate applicability to multiple boundary configurations
  (e.g., slit vs. plane plate).
"""

import numpy as np
import matplotlib.pyplot as plt
import meshio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =========================================================
# Configuration
# =========================================================

# Geometry candidates (keep both as documentation)
MESH_PATH = "./mesh/slit.step.msh"
# MESH_PATH = "./mesh/flatplate.step.msh"

# Receiver grid
NX_REC, NY_REC = 20, 20
x_rec_span = (-0.38, 0.38)
y_rec_span = (-0.38, 0.38)
z_m = 0.01

# Source distribution (Fibonacci spiral)
N_SRC = 100
x_src_span = (-0.75, 0.75)
y_src_span = (-0.75, 0.75)
z_src = 0.10
Z_OFFSET = 0.002  # slight offset to avoid overlap with boundary

# Visualization
Z_LIM = (-0.2, 0.8)
TITLE = "Measurement layout (Fibonacci spiral sources)"


# =========================================================
# Fibonacci spiral sampling
# =========================================================
def fibonacci_spiral_sources(N, xspan, yspan, z_fixed, z_offset):
    """
    Generate N points using Fibonacci spiral sampling
    on a disk and scale to fit within a rectangular region.

    Returns
    -------
    (N, 3) ndarray
        Source coordinates [x, y, z]
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    R = min(abs(xspan[0]), abs(xspan[1]), abs(yspan[0]), abs(yspan[1]))

    xs, ys = [], []
    for i in range(N):
        r = R * np.sqrt((i + 0.5) / N)
        theta = 2 * np.pi * (i / phi)
        xs.append(r * np.cos(theta))
        ys.append(r * np.sin(theta))

    xs = np.asarray(xs) / R * max(abs(xspan[0]), abs(xspan[1]))
    ys = np.asarray(ys) / R * max(abs(yspan[0]), abs(yspan[1]))
    zs = np.full_like(xs, z_fixed + z_offset)

    return np.column_stack([xs, ys, zs])


# =========================================================
# Main
# =========================================================
def main():

    # -----------------------------------------------------
    # Receiver grid
    # -----------------------------------------------------
    xv = np.linspace(*x_rec_span, NX_REC)
    yv = np.linspace(*y_rec_span, NY_REC)
    xxr, yyr = np.meshgrid(xv, yv, indexing="xy")
    zzr = np.full_like(xxr, z_m)

    rec_points = np.stack([xxr.ravel(), yyr.ravel(), zzr.ravel()], axis=-1)
    print(f"[INFO] Receivers: {rec_points.shape[0]} points")

    # -----------------------------------------------------
    # Source points
    # -----------------------------------------------------
    src_points = fibonacci_spiral_sources(
        N_SRC, x_src_span, y_src_span,
        z_fixed=z_src, z_offset=Z_OFFSET
    )
    print(f"[INFO] Sources: {src_points.shape[0]} (Fibonacci spiral)")

    # -----------------------------------------------------
    # Load mesh (mm -> m)
    # -----------------------------------------------------
    mesh = meshio.read(MESH_PATH)
    mesh.points *= 1e-3

    # Mesh summary
    print("\n[Mesh information]")
    print(f"  Number of nodes: {mesh.points.shape[0]}")
    for cell in mesh.cells:
        print(f"  {cell.type:10s}: {cell.data.shape[0]}")

    # -----------------------------------------------------
    # Figure setup
    # -----------------------------------------------------
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(8.5, 5.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(TITLE, pad=10)

    # -----------------------------------------------------
    # Draw mesh (faces first)
    # -----------------------------------------------------
    face_polys = []
    for cell in mesh.cells:
        if cell.type in ("triangle", "quad"):
            face_polys.extend(mesh.points[idx] for idx in cell.data)

    if face_polys:
        poly = Poly3DCollection(
            face_polys,
            facecolors="#d9d9d9",
            edgecolors="none",
            linewidths=0.1,
            alpha=0.25,
        )
        try:
            poly.set_zsort("min")
        except Exception:
            pass
        ax.add_collection3d(poly)

    # -----------------------------------------------------
    # Mesh edges (if present)
    # -----------------------------------------------------
    for cell in mesh.cells:
        if cell.type == "line":
            for a, b in cell.data:
                ax.plot(
                    mesh.points[[a, b], 0],
                    mesh.points[[a, b], 1],
                    mesh.points[[a, b], 2],
                    lw=0.6,
                    color="k",
                    alpha=0.55,
                )

    # -----------------------------------------------------
    # Receivers
    # -----------------------------------------------------
    ax.scatter(
        rec_points[:, 0], rec_points[:, 1], rec_points[:, 2],
        s=22, c="black", edgecolors="white", marker="o",
        linewidths=0.8, depthshade=False,
        label=f"Receivers (N={rec_points.shape[0]})",
    )

    # -----------------------------------------------------
    # Sources
    # -----------------------------------------------------
    ax.scatter(
        src_points[:, 0], src_points[:, 1], src_points[:, 2],
        s=46, c="green", edgecolors="black", marker="x",
        linewidths=1.4, depthshade=False,
        label=f"Sources (N={src_points.shape[0]})",
    )

    # -----------------------------------------------------
    # Axes & limits
    # -----------------------------------------------------
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(frameon=False, loc="upper right")

    all_pts = np.vstack([mesh.points, rec_points, src_points])
    xyz_min, xyz_max = all_pts.min(axis=0), all_pts.max(axis=0)
    L = (xyz_max - xyz_min).max()
    mid = 0.5 * (xyz_max + xyz_min)

    ax.set_xlim(mid[0] - L / 2, mid[0] + L / 2)
    ax.set_ylim(mid[1] - L / 2, mid[1] + L / 2)
    ax.set_zlim(*Z_LIM)

    ax.grid(False)
    plt.show()


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    main()
