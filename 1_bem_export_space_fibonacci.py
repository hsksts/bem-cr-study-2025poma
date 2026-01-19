#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_bem_export_space_fibonacci.py

BEM -> spatial pressure snapshot exporter (PoMA reference).

This script computes spatial pressure snapshots using BEM and exports
direct and reflected components (p_dir, p_ref) in the spatial domain.
No wavenumber-domain processing is performed here.

Key features:
- Deterministic source placement using Fibonacci spiral sampling
- Compatible with both slit and flat-plate geometries
- Spatial snapshots are saved for subsequent Cr estimation

Pipeline position:
    0_generate_mesh.py
 -> 1_bem_export_space_fibonacci.py
 -> 2_estimate_cr_from_space.py

Notes:
- Only the first source is visualized for monitoring.
- Geometry switch is intentionally left as commented alternatives
  to indicate multiple applicable boundary configurations.
"""

import numpy as np
from numpy.linalg import norm, solve
from numpy import pi, exp
import matplotlib.pyplot as plt


# =========================================================
# User settings
# =========================================================

# Geometry candidates (keep both to indicate available cases)
# MESH_PATH = "./mesh/slit.step.msh"
MESH_PATH = "./mesh/flatplate.step.msh"

MESH_UNIT = "mm"

# Physical parameters
f = 3400.0
rho, c = 1.21, 343.0
omega = 2 * pi * f
k = omega / c
Z0 = rho * c

# Ground-truth reflection coefficient (for validation)
Cr_true = 0.99
Y = (Cr_true - 1.0) / (Cr_true + 1.0) / Z0

# Receiver grid (plane parallel to boundary)
NX_REC, NY_REC = 20, 20
x_rec_span = (-0.38, 0.38)
y_rec_span = (-0.38, 0.38)
z_m = 0.01

# Source distribution (Fibonacci spiral)
Nsrc = 100
x_src_min, x_src_max = -0.75, 0.75
y_src_min, y_src_max = -0.75, 0.75
z_src = 0.10

OUT_NPZ = (
    "space_snapshot_f{f:.0f}Hz_"
    "Nrec{Nrec}_Nsrc{Nsrc}_"
    + MESH_PATH.replace("/", "_")
    + ".npz"
)


# =========================================================
# Fibonacci spiral source generator
# =========================================================
def fibonacci_spiral_sources(N, x_min, x_max, y_min, y_max, z_fixed):
    """
    Generate N source positions using a Fibonacci spiral,
    scaled to fit inside a rectangular region.

    Returns
    -------
    (N, 3) ndarray
        Source coordinates [x, y, z]
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    R = min(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

    xs, ys = [], []
    for i in range(N):
        r = R * np.sqrt((i + 0.5) / N)
        theta = 2 * pi * (i / phi)
        xs.append(r * np.cos(theta))
        ys.append(r * np.sin(theta))

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    xs *= max(abs(x_min), abs(x_max)) / R
    ys *= max(abs(y_min), abs(y_max)) / R
    zs = np.full(N, z_fixed)

    return np.stack([xs, ys, zs], axis=-1)


# =========================================================
# Mesh loading utility
# =========================================================
def load_gmsh_as_triangles(path):
    """
    Load a Gmsh surface mesh and convert quads to triangles.
    """
    import meshio

    m = meshio.read(path)
    nodes = np.asarray(m.points[:, :3], dtype=np.float64)

    tri_list = []
    for blk in m.cells:
        if "triangle" in blk.type:
            tri_list.append(blk.data[:, :3])
        elif "quad" in blk.type:
            q = blk.data[:, :4]
            tri_list.append(q[:, [0, 1, 2]])
            tri_list.append(q[:, [0, 2, 3]])

    if not tri_list:
        raise ValueError("No surface elements found in mesh.")

    tris = np.vstack(tri_list).astype(np.int32)
    return nodes, tris


# =========================================================
# Load mesh and build geometry
# =========================================================
nodes, elem_idx = load_gmsh_as_triangles(MESH_PATH)
if MESH_UNIT.lower() == "mm":
    nodes *= 1e-3

p = nodes[elem_idx]
centers = p.mean(axis=1)

e1 = p[:, 1] - p[:, 0]
e2 = p[:, 2] - p[:, 0]
n_raw = np.cross(e1, e2)
n_len = norm(n_raw, axis=1)

areas = 0.5 * np.maximum(n_len, 1e-20)
normals = n_raw / np.maximum(n_len, 1e-20)[:, None]
normals *= -1.0  # enforce outward normal


# =========================================================
# BEM system matrix (constant admittance)
# =========================================================
Rji = centers[np.newaxis, :, :] - centers[:, np.newaxis, :]
rji = norm(Rji, axis=-1)

rhat = Rji / (rji[..., None] + 1e-12)
Gmat = exp(1j * k * rji) / (4 * pi * np.maximum(rji, 1e-12))
np.fill_diagonal(Gmat, 0.0)

dotn = np.einsum("ijk,jk->ij", rhat, normals)
dG = ((1j * k - 1.0 / (rji + 1e-12)) * Gmat) * dotn
np.fill_diagonal(dG, 0.0)

A = (dG - 1j * k * Z0 * Y * Gmat) * areas[np.newaxis, :]
A[np.diag_indices_from(A)] += 0.5


# =========================================================
# Linear solver
# =========================================================
try:
    from scipy.linalg import lu_factor, lu_solve
    LU = lu_factor(A)

    def solve_A(rhs):
        return lu_solve(LU, rhs)

except Exception:
    def solve_A(rhs):
        return solve(A, rhs)


# =========================================================
# Receiver grid
# =========================================================
xv = np.linspace(*x_rec_span, NX_REC)
yv = np.linspace(*y_rec_span, NY_REC)
xxr, yyr = np.meshgrid(xv, yv, indexing="xy")
zzr = np.full_like(xxr, z_m)

rec_points = np.stack([xxr.ravel(), yyr.ravel(), zzr.ravel()], axis=-1)
Nrec = rec_points.shape[0]


# =========================================================
# Source loop
# =========================================================
src_points = fibonacci_spiral_sources(
    Nsrc,
    x_src_min, x_src_max,
    y_src_min, y_src_max,
    z_src
)

print(f"[INFO] Number of sources: {Nsrc} (Fibonacci spiral)")

# Precompute receiver kernels
R_obs = rec_points[:, None, :] - centers[None, :, :]
r_obs = norm(R_obs, axis=-1)
rhat_o = R_obs / (r_obs[..., None] + 1e-12)

G_obs = exp(1j * k * r_obs) / (4 * pi * np.maximum(r_obs, 1e-12))
dotn_o = np.einsum("ijk,jk->ij", rhat_o, normals)
dG_obs = ((1j * k - 1.0 / (r_obs + 1e-12)) * G_obs) * dotn_o

Acol = areas[None, :]
Wp = dG_obs * Acol
Wv = (-1j * k * Z0) * (G_obs * Acol)

p_dir = np.zeros((Nsrc, Nrec), dtype=np.complex64)
p_ref = np.zeros((Nsrc, Nrec), dtype=np.complex64)

for si, src in enumerate(src_points):
    r_sc = norm(centers - src, axis=1)
    b = exp(1j * k * r_sc) / (4 * pi * np.maximum(r_sc, 1e-12))

    p_b = solve_A(b)
    v_b = Y * p_b

    p_ref[si] = (Wp @ p_b + Wv @ v_b).astype(np.complex64)

    r_sr = norm(rec_points - src, axis=1)
    p_dir[si] = exp(1j * k * r_sr) / (4 * pi * np.maximum(r_sr, 1e-12))

    # Monitoring plot (first source only)
    if si == 0:
        P = np.abs(p_dir[si] + p_ref[si]).reshape(NY_REC, NX_REC)
        plt.figure(figsize=(6, 5))
        plt.pcolormesh(xxr, yyr, P, shading="auto", cmap="Reds")
        plt.colorbar(label="|p|")
        plt.title("Monitoring: |p_total| (source #0)")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.gca().set_aspect("equal")
        plt.show()


print(f"[BEM] Exporting p_dir {p_dir.shape}, p_ref {p_ref.shape}")


# =========================================================
# Save snapshot
# =========================================================
out_path = OUT_NPZ.format(f=f, Nrec=Nrec, Nsrc=Nsrc)

np.savez_compressed(
    out_path,
    p_dir=p_dir,
    p_ref=p_ref,
    rec_points=rec_points,
    src_points=src_points,
    f=np.array(f),
    k=np.array(k),
    rho=np.array(rho),
    c=np.array(c),
    Z0=np.array(Z0),
    z_m=np.array(z_m),
    mesh_path=np.array(MESH_PATH),
    note=np.array(
        "Spatial pressure snapshot with Fibonacci sources. "
        "Use 2_estimate_cr_from_space.py for k-space processing."
    ),
)

print(f"[SAVE] {out_path}")
