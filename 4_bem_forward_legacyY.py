#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight BEM (scalar admittance Y) for baseline comparison.

This script provides a *legacy / baseline* BEM solver using a
spatially-uniform (scalar) boundary admittance Y.

Purpose
-------
- Provide a reference solution for comparison with:
    - k-space admittance BEM (proposed method)
    - Cr-based nonlocal boundary operators
- Useful for sanity checks and visual intuition.

Features
--------
- Load surface mesh (.msh, Gmsh v2/v4)
- Assemble BEM with scalar admittance Y
- Solve boundary unknowns (pressure, velocity)
- Evaluate XZ (y=0) and YZ (x=0) cross-sections
- 2D plots + 3D orthogonal cross-slices
- Overlay slices on geometry mesh
- Export results to NPZ (postproc_view.py compatible)

Notes
-----
- Boundary condition:
      ∂p/∂n = - i k Z0 Y p
- Here Y is *scalar* (local admittance).
- This script intentionally ignores k-space structure.

IMPORTANT (Alignment with proposed pipeline)
--------------------------------------------
To enable direct Cosine similarity / MSE comparison in postproc_view.py,
the post-processing grid is aligned with `3_bem_forward_proposed.py`:
- Uses pitch-based sampling (DX_SLICE / DY_SLICE)
- Uses fixed NZ_YZ for YZ z-samples
- Exports keys: xv, zv, yv, zv_yz, p_total_xz, p_total_yz, MESH_PATH, title_tag, clim/xlim/ylim/zlim

Author
------
satoshihoshika (baseline reference)
"""

from __future__ import annotations

import os
import numpy as np
from numpy.linalg import solve, norm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from numpy import pi
import meshio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =========================================================
# Configuration
# =========================================================

# --- Physical parameters ---
rho = 1.21
c   = 343.0
f   = 3400.0
omega = 2 * pi * f
k = omega / c
Z0 = rho * c

# --- Scalar reflection coefficient (baseline) ---
Cr = 0.99
# Admittance Y = (Cr - 1) / (Cr + 1) / Z0
Y = (Cr - 1.0) / (Cr + 1.0 + 1e-24) / Z0

# --- Point source ---
x_src = np.array([0.0, 0.0, 0.4], dtype=float)

# --- Geometry mesh (surface only) ---
# MESH_PATH = "./mesh/slit.step.msh"
MESH_PATH = "./mesh/flatplate.step.msh"
MESH_UNIT = "mm"   # "mm" or "m"

# --- Post-processing region (match proposed) ---
POSTPROC_XLIM = (-0.75, 0.75)
POSTPROC_YLIM = (-0.75, 0.75)
POSTPROC_ZLIM = (0.0, 1.0)     # proposed is (0,1). legacy baseline originally used (0.01,1.0)

# --- Sampling resolution (match proposed) ---
# pitch-based sampling for X and Y
DX_SLICE = 0.01   # [m] x pitch in XZ
DY_SLICE = 0.01   # [m] y pitch in YZ
# z sampling for YZ plane
NZ_YZ    = 100    # number of z samples on YZ plane

# --- Visualization defaults ---
POSTPROC_CMAP = "Reds"
POSTPROC_CLIM = (0.0, 1.0)
Z_LIM_MESH    = (-0.2, 0.8)

PLANE_ALPHA_XZ = 1.0
PLANE_ALPHA_YZ = 1.0

# --- Export ---
EXPORT_DIR = "postproc_data"
EXPORT_TAG = "legacyY"


# =========================================================
# Helpers
# =========================================================
def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def linspace_by_pitch(vmin: float, vmax: float, dv: float) -> np.ndarray:
    """
    Create linspace with approx pitch dv, ensuring >= 2 samples.

    This matches the approach used in 3_bem_forward_proposed.py.
    """
    L = float(vmax) - float(vmin)
    n = int(np.floor(L / max(dv, 1e-12))) + 1
    n = max(n, 2)
    return np.linspace(vmin, vmax, n, dtype=np.float64)


# =========================================================
# Mesh loading (triangulate if needed)
# =========================================================
def load_gmsh_as_triangles(path: str, unit: str = "mm"):
    """
    Load a Gmsh .msh file and return only triangular faces.

    Quad faces are internally split into two triangles.

    Returns
    -------
    nodes : (N,3) float64 [m]
    tris  : (M,3) int32
    """
    mesh = meshio.read(path)
    nodes = np.asarray(mesh.points[:, :3], dtype=float)

    if unit.lower() == "mm":
        nodes *= 1e-3

    tri_list = []

    for cell in mesh.cells:
        if cell.type == "triangle":
            tri_list.append(cell.data.astype(np.int32))
        elif cell.type == "quad":
            q = cell.data.astype(np.int32)
            tri_list.append(q[:, [0, 1, 2]])
            tri_list.append(q[:, [0, 2, 3]])

    if not tri_list:
        raise RuntimeError("No surface triangles/quads found in mesh.")

    tris = np.vstack(tri_list)
    return nodes, tris


# =========================================================
# Geometry utilities
# =========================================================
def triangle_geometry(nodes: np.ndarray, tris: np.ndarray):
    """
    Compute centers, areas, and unit normals for triangular facets.
    """
    p = nodes[tris]
    centers = p.mean(axis=1)

    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    n_raw = np.cross(e1, e2)

    n_len = norm(n_raw, axis=1)
    areas = 0.5 * n_len
    normals = n_raw / np.maximum(n_len, 1e-20)[:, None]

    return centers, areas, normals


# =========================================================
# BEM assembly (scalar Y)
# =========================================================
def assemble_BEM_matrix(centers, normals, areas, k, Y, Z0):
    """
    Assemble BEM matrix for scalar admittance boundary condition.

    A_ij = ∂G/∂n_j - i k Z0 Y G
    """
    C = centers
    R = C[None, :, :] - C[:, None, :]
    r = norm(R, axis=-1)

    rhat = R / (r[..., None] + 1e-12)
    G = np.exp(1j * k * r) / (4 * pi * np.maximum(r, 1e-12))
    np.fill_diagonal(G, 0.0)

    dot_n = np.einsum("ijk,jk->ij", rhat, normals)
    dG = ((1j * k - 1.0 / np.maximum(r, 1e-12)) * G) * dot_n
    np.fill_diagonal(dG, 0.0)

    A = (dG - 1j * k * Z0 * Y * G) * areas[None, :]
    A[np.diag_indices_from(A)] += 0.5

    return A


def solve_boundary(centers, normals, areas, k, Z0, Y, x_src):
    """
    Solve boundary pressure and velocity.
    """
    A = assemble_BEM_matrix(centers, normals, areas, k, Y, Z0)

    r_src = norm(centers - x_src, axis=1)
    b = np.exp(1j * k * r_src) / (4 * pi * np.maximum(r_src, 1e-12))

    p_b = solve(A, b)
    v_b = Y * p_b

    return p_b, v_b


# =========================================================
# Field evaluation
# =========================================================
def evaluate_field(centers, normals, areas, k, Z0,
                   p_b, v_b, x_src, obs):
    """
    Evaluate scattered and total field at observation points.

    Returns
    -------
    p_r   : scattered/reflected field
    p_tot : total field (direct + scattered)
    """
    R = obs[:, None, :] - centers[None, :, :]
    r = norm(R, axis=-1)
    rhat = R / (r[..., None] + 1e-12)

    G = np.exp(1j * k * r) / (4 * pi * np.maximum(r, 1e-12))
    dot_n = np.einsum("ijk,jk->ij", rhat, normals)
    dG = ((1j * k - 1.0 / np.maximum(r, 1e-12)) * G) * dot_n

    p_r = ((dG * p_b[None, :] - 1j * k * Z0 * G * v_b[None, :])
           * areas[None, :]).sum(axis=1)

    r_src = norm(obs - x_src, axis=1)
    p_dir = np.exp(1j * k * r_src) / (4 * pi * np.maximum(r_src, 1e-12))

    return p_r, p_dir + p_r


# =========================================================
# Visualization helpers (3D cross-slices) - kept for compatibility
# =========================================================
def set_equal_axes_3d(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ], dtype=float)

    centers = limits.mean(axis=1)
    spans = limits[:, 1] - limits[:, 0]
    L = max(spans.max(), 1e-12)

    ax.set_xlim3d(centers[0] - L/2, centers[0] + L/2)
    ax.set_ylim3d(centers[1] - L/2, centers[1] + L/2)
    ax.set_zlim3d(centers[2] - L/2, centers[2] + L/2)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


# =========================================================
# Main
# =========================================================
def main():

    # ---- Load mesh ----
    if not os.path.exists(MESH_PATH):
        raise FileNotFoundError(f"MESH_PATH not found: {MESH_PATH}")

    nodes, tris = load_gmsh_as_triangles(MESH_PATH, unit=MESH_UNIT)
    centers, areas, normals = triangle_geometry(nodes, tris)

    # Enforce outward (downward) normals (consistent with proposed scripts)
    normals[normals[:, 2] > 0] *= -1.0

    # ---- Boundary solve ----
    p_b, v_b = solve_boundary(
        centers, normals, areas,
        k, Z0, Y, x_src
    )

    # =====================================================
    # Post-processing grids (MATCH proposed)
    # =====================================================

    # ---- XZ plane (y=0) ----
    xv = linspace_by_pitch(POSTPROC_XLIM[0], POSTPROC_XLIM[1], DX_SLICE)
    zv = linspace_by_pitch(POSTPROC_ZLIM[0], POSTPROC_ZLIM[1], DX_SLICE)
    xx, zz = np.meshgrid(xv, zv, indexing="xy")
    obs_xz = np.stack(
        [xx.ravel(), np.zeros_like(xx).ravel(), zz.ravel()],
        axis=-1
    )

    _, p_total_xz = evaluate_field(
        centers, normals, areas,
        k, Z0, p_b, v_b, x_src, obs_xz
    )
    p_total_xz = p_total_xz.reshape(zz.shape)

    # ---- YZ plane (x=0) ----
    yv = linspace_by_pitch(POSTPROC_YLIM[0], POSTPROC_YLIM[1], DY_SLICE)
    zv_yz = np.linspace(POSTPROC_ZLIM[0], POSTPROC_ZLIM[1], NZ_YZ, dtype=np.float64)
    yy, zz_yz = np.meshgrid(yv, zv_yz, indexing="xy")
    obs_yz = np.stack(
        [np.zeros_like(yy).ravel(), yy.ravel(), zz_yz.ravel()],
        axis=-1
    )

    _, p_total_yz = evaluate_field(
        centers, normals, areas,
        k, Z0, p_b, v_b, x_src, obs_yz
    )
    p_total_yz = p_total_yz.reshape(zz_yz.shape)

    # =====================================================
    # Export (postproc_view.py compatible, shape-aligned)
    # =====================================================
    ensure_dir(EXPORT_DIR)

    base = os.path.splitext(os.path.basename(MESH_PATH))[0]
    out = os.path.join(EXPORT_DIR, f"postproc_legacyY__{base}.npz")

    np.savez_compressed(
        out,
        # grids
        xv=xv, zv=zv, yv=yv, zv_yz=zv_yz,
        # fields
        p_total_xz=p_total_xz,
        p_total_yz=p_total_yz,
        # meta
        MESH_PATH=os.path.abspath(MESH_PATH),
        mesh_unit=MESH_UNIT,
        title_tag=f"legacyY, Cr={Cr:.3f}",
        x_src=x_src,
        f=np.array(f, dtype=float),
        k=np.array(k, dtype=float),
        Z0=np.array(Z0, dtype=float),
        Y=np.array(Y, dtype=float),
        # visualization meta (for viewer)
        cmap=np.array(POSTPROC_CMAP),
        clim=np.array(POSTPROC_CLIM, dtype=float),
        xlim=np.array(POSTPROC_XLIM, dtype=float),
        ylim=np.array(POSTPROC_YLIM, dtype=float),
        zlim=np.array(POSTPROC_ZLIM, dtype=float),
    )

    print(f"[EXPORT] {out}")


if __name__ == "__main__":
    main()
