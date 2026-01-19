#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_bem_forward_proposed.py

Forward BEM simulation using a *standard 2D Fourier transform*
on the boundary plane (no RKHS / no reproducing kernel).

Purpose:
- Validate estimated wavenumber-domain reflection coefficient Cr
  by reconstructing the sound field with BEM.
- This script corresponds to the "forward validation" step in PoMA.

Main features:
- Load Cr (nk × nk) and k-vectors from NPZ
- Construct standard 2D Fourier basis on boundary plane (z = 0)
- Apply whitening: F_tilde^H W F_tilde ≈ I
- Map k-space admittance Y_k → boundary operator
- Solve BEM with mixed boundary condition
- Visualize XZ / YZ sections and 3D orthogonal cross-slices
- Export post-processing data to NPZ

Notes:
- Cr is used *as-is* (no transpose, no scaling).
- Convention:  Pr = Cr @ Pi
- flatplate / slit geometries can be switched via file paths.

Author: satoshihoshika
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve, eigh

from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =========================================================
# CONFIG
# =========================================================

# ---- Estimated Cr (from step 03) ----
# CR_FILE = "Cr_from_space_estimated__space_snapshot_f3400Hz_Nrec400_Nsrc100_._mesh_flatplate.step.msh__nk2401.npz"
CR_FILE = "Cr_from_space_estimated__space_snapshot_f3400Hz_Nrec400_Nsrc100_._mesh_slit.step.msh__nk2401.npz"

# ---- Mesh for visualization only (not used in BEM solve) ----
# MESH_PATH = "mesh/slit.step.msh"
MESH_PATH = "mesh/flatplate.step.msh"
Z_LIM_MESH = (-0.2, 1.0)

# ---- Physical parameters ----
rho = 1.21
c   = 343.0
Z0  = rho * c

# ---- Plate discretization (boundary elements; structured plate) ----
MESH_TYPE = "tri"          # "tri" or "quad"
PLATE_SIZE = 1.5
N_BASE = 75
AUTO_BALANCE_TRI = True

# ---- Whitening / conditioning ----
EIG_EPS = 1e-10
USE_RANK_TRIM = True
RANK_TRIM_REL = 1e-9

# ---- Weight handling ----
PREFER_NPZ_WEIGHTS = False

# ---- Debug options ----
OVERRIDE_CR_TO_ALPHA_I = None     # e.g. 0.0, 0.6, 1.0
FORCE_NO_TRANSPOSE_OR_SCALE = True

# ---- Post-processing export ----
EXPORT_DIR = "postproc_data"
EXPORT_TAG = None

POSTPROC_CMAP = "Reds"
POSTPROC_CLIM = (0.0, 1.0)

POSTPROC_XLIM = (-0.75, 0.75)
POSTPROC_YLIM = (-0.75, 0.75)
POSTPROC_ZLIM = (0.0, 1.0)

# sampling pitch for slices
DX_SLICE = 0.01   # [m]
DY_SLICE = 0.01   # [m]
NZ_YZ    = 100    # z samples on YZ plane

PLANE_ALPHA_XZ = 0.98
PLANE_ALPHA_YZ = 0.85


# =========================================================
# Helpers
# =========================================================
def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def linspace_by_pitch(vmin: float, vmax: float, dv: float) -> np.ndarray:
    """Create linspace with approx pitch dv, ensuring >=2 samples."""
    L = float(vmax) - float(vmin)
    n = int(np.floor(L / max(dv, 1e-12))) + 1
    n = max(n, 2)
    return np.linspace(vmin, vmax, n, dtype=np.float64)

def mesh_stats_from_meshio(mesh) -> dict:
    """Return a small dict of cell type counts for printing."""
    counts = {}
    for cell in mesh.cells:
        ctype = cell.type
        n = int(cell.data.shape[0])
        counts[ctype] = counts.get(ctype, 0) + n
    return counts


# =========================================================
# Plate mesh generation (structured)
# =========================================================
def generate_square_plate_mesh(N: int, size: float):
    x = np.linspace(-size/2, size/2, N, dtype=np.float64)
    y = np.linspace(-size/2, size/2, N, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx).ravel()], axis=-1)

    quads = []
    for i in range(N - 1):
        base = i * N
        for j in range(N - 1):
            n0 = base + j
            n1 = n0 + 1
            n2 = n0 + N
            n3 = n2 + 1
            quads.append([n0, n1, n3, n2])  # ccw
    return nodes.astype(np.float64), np.asarray(quads, dtype=np.int32), xx, yy

def generate_triangle_plate_mesh(N: int, size: float):
    x = np.linspace(-size/2, size/2, N, dtype=np.float64)
    y = np.linspace(-size/2, size/2, N, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx).ravel()], axis=-1)

    tris = []
    for i in range(N - 1):
        for j in range(N - 1):
            n0 = i * N + j
            n1 = n0 + 1
            n2 = n0 + N
            n3 = n2 + 1
            tris.append([n0, n1, n3])  # ccw
            tris.append([n0, n3, n2])  # ccw
    return nodes.astype(np.float64), np.asarray(tris, dtype=np.int32), xx, yy


# ---- effective resolution ----
if MESH_TYPE == "tri" and AUTO_BALANCE_TRI:
    N_eff = int(np.floor((N_BASE - 1) / np.sqrt(2.0))) + 1
    N_eff = max(N_eff, 3)
else:
    N_eff = N_BASE

if MESH_TYPE == "tri":
    nodes, elem_idx, Xgrid, Ygrid = generate_triangle_plate_mesh(N_eff, PLATE_SIZE)
else:
    nodes, elem_idx, Xgrid, Ygrid = generate_square_plate_mesh(N_eff, PLATE_SIZE)

num_elem = int(len(elem_idx))

p = nodes[elem_idx]              # (M, 3/4, 3)
centers = p.mean(axis=1)         # (M, 3)

# ---- areas & normals (downward) ----
if MESH_TYPE == "tri":
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    normal_raw = np.cross(e1, e2)
    norm_len = norm(normal_raw, axis=1)
    areas = 0.5 * norm_len
    normals = normal_raw / np.maximum(norm_len, 1e-20)[:, None]
else:
    v1 = p[:, 1] - p[:, 0]
    v2 = p[:, 3] - p[:, 0]
    normal_raw = np.cross(v1, v2)
    norm_len = norm(normal_raw, axis=1)
    areas = norm_len
    normals = normal_raw / np.maximum(norm_len, 1e-20)[:, None]

# enforce outward downward (z<0)
normals[normals[:, 2] > 0] *= -1.0


# =========================================================
# Load Cr and k-vectors
# =========================================================
if not os.path.exists(CR_FILE):
    raise FileNotFoundError(f"CR_FILE not found: {CR_FILE}")

dat = np.load(CR_FILE, allow_pickle=True)

Cr_mat = dat["Cr"].astype(np.complex128)
kx = dat["kx_vec"].astype(np.float64)
ky = dat["ky_vec"].astype(np.float64)

nk = int(len(kx))
if ky.shape != (nk,):
    raise ValueError("ky_vec shape mismatch.")
if Cr_mat.shape != (nk, nk):
    raise ValueError(f"Cr shape mismatch: expected ({nk},{nk}) got {Cr_mat.shape}")

k = float(dat["k"]) if "k" in dat else None
f = float(dat["f"]) if "f" in dat else None
if k is None or f is None:
    raise ValueError("NPZ must contain 'k' and 'f' for consistency.")

omega = 2.0 * np.pi * f

# kz
if "kz_vec" in dat:
    kz = dat["kz_vec"].astype(np.float64)
else:
    kz = np.sqrt(np.maximum(k**2 - (kx**2 + ky**2), 0.0))

# ---- optional weights ----
wk = None
if PREFER_NPZ_WEIGHTS:
    for key in ("wk", "wk_disk", "weights_k"):
        if key in dat:
            cand = np.asarray(dat[key], dtype=np.float64)
            if cand.shape == (nk,):
                wk = cand
                print(f"[INFO] Using NPZ weights: {key}")
                break

if wk is None:
    wk = (2.0 * np.pi * (k**2) / nk) * (kz / k)

# ---- debug override ----
if OVERRIDE_CR_TO_ALPHA_I is not None:
    alpha = float(OVERRIDE_CR_TO_ALPHA_I)
    Cr_mat = alpha * np.eye(nk, dtype=np.complex128)
    print(f"[DEBUG] Overriding Cr := {alpha} * I")

# ---- enforce no transpose/scale ----
if FORCE_NO_TRANSPOSE_OR_SCALE:
    pass


# =========================================================
# Standard 2D Fourier transform on boundary
# =========================================================
x_c = centers[:, 0]
y_c = centers[:, 1]

# Plane-wave basis on z=0: exp(-i(kx x + ky y))
F = (1.0 / (2.0 * np.pi)) * np.exp(-1j * (np.outer(kx, x_c) + np.outer(ky, y_c)))


# =========================================================
# Whitening: F_tilde^H W F_tilde ≈ I
# =========================================================
H = F.conj().T @ (F * wk[:, None])         # (M, M)
lam, U = eigh(H)

lam_max = float(np.max(lam)) if lam.size else 0.0

if USE_RANK_TRIM:
    keep = lam > max(EIG_EPS, RANK_TRIM_REL * lam_max)
    if not np.any(keep):
        raise RuntimeError("All eigenvalues are below threshold; relax EIG_EPS/RANK_TRIM_REL.")
    U_k = U[:, keep]
    lam_k = lam[keep]
    H_inv_sqrt = (U_k * (1.0 / np.sqrt(lam_k))) @ U_k.conj().T
else:
    lam_clip = np.clip(lam, EIG_EPS, None)
    H_inv_sqrt = (U * (1.0 / np.sqrt(lam_clip))) @ U.conj().T

F_tilde = F @ H_inv_sqrt

I_test = F_tilde.conj().T @ (F_tilde * wk[:, None])
err_I = norm(I_test - np.eye(I_test.shape[0])) / max(1.0, norm(np.eye(I_test.shape[0])))
print(f"[CHECK] whitening error (relative) = {err_I:.3e}  (M={I_test.shape[0]}, nk={nk})")


# =========================================================
# Boundary admittance operator
# =========================================================
I_k = np.eye(nk, dtype=np.complex128)

# agreed sign/scale: B0 = diag(kz/(Z0*k))
B0 = np.diag((kz / (Z0 * k)).astype(np.complex128))
Y_k = B0 @ (Cr_mat - I_k) @ np.linalg.inv(Cr_mat + I_k)     # (nk, nk)

# Project to boundary: Y_b = F_tilde^H [ W (Y_k F_tilde) ]
YF = Y_k @ F_tilde
Y_boundary = F_tilde.conj().T @ (wk[:, None].astype(np.complex128) * YF)    # (M, M)


# =========================================================
# BEM assembly and solve
# =========================================================
R = centers[:, None, :] - centers[None, :, :]   # (M, M, 3)
r = norm(R, axis=-1)
inv_r = 1.0 / (r + 1e-12)

G = np.exp(1j * k * r) * inv_r / (4.0 * np.pi)

# dG/dn' (normal at source element)
dot_n = np.einsum("ijk,jk->ij", R, normals) * inv_r
dG = ((1j * k - inv_r) * G) * dot_n

np.fill_diagonal(G, 0.0)
np.fill_diagonal(dG, 0.0)

A = (dG - 1j * k * (G @ Y_boundary)) * areas[None, :]
A[np.diag_indices(num_elem)] += 0.5

x_src = np.array([0.0, 0.0, 0.4], dtype=np.float64)
r_src = norm(centers - x_src, axis=1)
b = np.exp(1j * k * r_src) / (4.0 * np.pi * np.maximum(r_src, 1e-12))

p_b = solve(A, b)                  # boundary pressure
v_b = Y_boundary @ p_b             # boundary velocity (admittance operator)


# =========================================================
# Field evaluation
# =========================================================
# ---- XZ plane (y=0) ----
xv = linspace_by_pitch(POSTPROC_XLIM[0], POSTPROC_XLIM[1], DX_SLICE)
zv = linspace_by_pitch(POSTPROC_ZLIM[0], POSTPROC_ZLIM[1], DX_SLICE)

xx, zz = np.meshgrid(xv, zv, indexing="xy")
obs_points = np.stack([xx.ravel(), np.zeros_like(xx).ravel(), zz.ravel()], axis=-1)

Ro = obs_points[:, None, :] - centers[None, :, :]
ro = norm(Ro, axis=-1)
inv_ro = 1.0 / (ro + 1e-12)

G_obs = np.exp(1j * k * ro) * inv_ro / (4.0 * np.pi)
dot_n_o = np.einsum("ijk,jk->ij", Ro, normals) * inv_ro
dG_obs = ((1j * k - inv_ro) * G_obs) * dot_n_o

Acol = areas[None, :]

# reflected/scattered
p_r = ((dG_obs * p_b[None, :] - 1j * rho * c * k * G_obs * v_b[None, :]) * Acol).sum(axis=1)

# direct field
r_src_obs = norm(obs_points - x_src, axis=1)
p_dir = np.exp(1j * k * r_src_obs) / (4.0 * np.pi * np.maximum(r_src_obs, 1e-12))

p_total = p_dir + p_r
p_ref = p_r


# ---- YZ plane (x=0) ----
yv = linspace_by_pitch(POSTPROC_YLIM[0], POSTPROC_YLIM[1], DY_SLICE)
zv_yz = np.linspace(POSTPROC_ZLIM[0], POSTPROC_ZLIM[1], NZ_YZ, dtype=np.float64)

yy, zz_yz = np.meshgrid(yv, zv_yz, indexing="xy")
obs_points_yz = np.stack([np.zeros_like(yy).ravel(), yy.ravel(), zz_yz.ravel()], axis=-1)

Ro_yz = obs_points_yz[:, None, :] - centers[None, :, :]
ro_yz = norm(Ro_yz, axis=-1)
inv_ro_yz = 1.0 / (ro_yz + 1e-12)

G_yz = np.exp(1j * k * ro_yz) * inv_ro_yz / (4.0 * np.pi)
dot_n_o_yz = np.einsum("ijk,jk->ij", Ro_yz, normals) * inv_ro_yz
dG_yz = ((1j * k - inv_ro_yz) * G_yz) * dot_n_o_yz

p_r_yz = ((dG_yz * p_b[None, :] - 1j * rho * c * k * G_yz * v_b[None, :]) * Acol).sum(axis=1)

r_src_yz = norm(obs_points_yz - x_src, axis=1)
p_dir_yz = np.exp(1j * k * r_src_yz) / (4.0 * np.pi * np.maximum(r_src_yz, 1e-12))

p_total_yz = p_dir_yz + p_r_yz
p_ref_yz = p_r_yz


# =========================================================
# 3D Cross-slice visualization helpers
# =========================================================
def _quad_slices(ax, X1, X2, A, fixed_axis, fixed_value, alpha=0.98, norm_obj=None, cmap="Reds"):
    """
    Plot a surface by splitting into 4 quads to avoid rendering issues.
    A is assumed shape (len(X1), len(X2)) in ij indexing.
    """
    n1, n2 = A.shape
    i1 = n1 // 2
    i2 = n2 // 2
    quads = [
        (slice(0, i1), slice(0, i2)),
        (slice(0, i1), slice(i2, n2)),
        (slice(i1, n1), slice(0, i2)),
        (slice(i1, n1), slice(i2, n2)),
    ]

    cmap_obj = plt.get_cmap(cmap)

    for s1, s2 in quads:
        G1, G2 = np.meshgrid(X1[s1], X2[s2], indexing="ij")
        if norm_obj is not None:
            colors = cmap_obj(norm_obj(A[s1, s2]))
        else:
            colors = None

        if fixed_axis == "x":
            X = np.full_like(G1, fixed_value, dtype=float)
            Y, Z = G1, G2
            ax.plot_surface(
                X, Y, Z,
                facecolors=colors,
                rstride=1, cstride=1,
                shade=False, linewidth=0,
                antialiased=False, alpha=alpha
            )
        elif fixed_axis == "y":
            Y = np.full_like(G1, fixed_value, dtype=float)
            X, Z = G1, G2
            ax.plot_surface(
                X, Y, Z,
                facecolors=colors,
                rstride=1, cstride=1,
                shade=False, linewidth=0,
                antialiased=False, alpha=alpha
            )
        else:
            raise ValueError("fixed_axis must be 'x' or 'y'.")

def _to_ij_abs(A, X, Z):
    """
    Convert array A to ij ordering as abs values, given grids X,Z.
    Accepts A shaped (len(Z), len(X)) in meshgrid('xy') convention.
    Returns A_ij shaped (len(X), len(Z)).
    """
    if A.shape == (len(Z), len(X)):
        return np.abs(A).T
    if A.shape == (len(X), len(Z)):
        return np.abs(A)
    raise ValueError(f"Array shape {A.shape} not compatible with X({len(X)}), Z({len(Z)})")

def plot_cross_slices_from_2d_grids(
    xv, zv, p_xz,
    yv, zv_yz, p_yz,
    title="3D Cross-slices (|p_total|)",
    alpha_xz=0.98, alpha_yz=0.90,
    cmap="Reds",
    clim=(0.0, 1.0),
    xlim=None, ylim=None, zlim=None,
):
    A_xz = _to_ij_abs(p_xz, xv, zv)       # (len(x), len(z))
    A_yz = _to_ij_abs(p_yz, yv, zv_yz)    # (len(y), len(z))

    norm_obj = Normalize(vmin=clim[0], vmax=clim[1])

    fig = plt.figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # XZ at y=0, so fixed_axis='y' with (X,Z)
    _quad_slices(ax, xv, zv, A_xz, fixed_axis="y", fixed_value=0.0,
                 alpha=alpha_xz, norm_obj=norm_obj, cmap=cmap)
    # YZ at x=0, so fixed_axis='x' with (Y,Z)
    _quad_slices(ax, yv, zv_yz, A_yz, fixed_axis="x", fixed_value=0.0,
                 alpha=alpha_yz, norm_obj=norm_obj, cmap=cmap)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    if xlim is not None:
        ax.set_xlim3d(*xlim)
    if ylim is not None:
        ax.set_ylim3d(*ylim)
    if zlim is not None:
        ax.set_zlim3d(*zlim)

    mappable = plt.cm.ScalarMappable(norm=norm_obj, cmap=plt.get_cmap(cmap))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.05)
    cbar.set_label("|p| [Pa]")

    ax.grid(False)
    plt.tight_layout()
    plt.show()

def plot_mesh_with_cross_slices(
    mesh_path,
    xv, zv, p_xz,
    yv, zv_yz, p_yz,
    title="Mesh + 3D cross-slices (|p|)",
    z_lim_mesh=(-0.2, 1.0),
    mesh_alpha=0.25,
    plane_alpha_xz=0.98,
    plane_alpha_yz=0.85,
    cmap="Reds",
    clim=(0.0, 1.0),
    xlim=None, ylim=None, zlim=None,
):
    import meshio  # local import (optional dependency)

    A_xz = _to_ij_abs(p_xz, xv, zv)
    A_yz = _to_ij_abs(p_yz, yv, zv_yz)

    mesh = meshio.read(mesh_path)  # assumed mm
    pts = mesh.points.astype(float) * 1e-3  # mm -> m

    fig = plt.figure(figsize=(9.0, 5.6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, pad=10)

    # faces
    face_polys = []
    for cell in mesh.cells:
        if cell.type in ("triangle", "quad"):
            face_polys.extend([pts[idx] for idx in cell.data])

    if face_polys:
        poly = Poly3DCollection(
            face_polys,
            facecolors="#d9d9d9",
            edgecolors="none",
            linewidths=0.2,
            alpha=mesh_alpha,
        )
        try:
            poly.set_zsort("min")
        except Exception:
            pass
        ax.add_collection3d(poly)

    # edges (if present)
    for cell in mesh.cells:
        if cell.type == "line":
            for a, b in cell.data:
                xa, ya, za = pts[a]
                xb, yb, zb = pts[b]
                ax.plot([xa, xb], [ya, yb], [za, zb], lw=0.6, color="k", alpha=0.6)

    norm_obj = Normalize(vmin=clim[0], vmax=clim[1])
    _quad_slices(ax, xv, zv, A_xz, fixed_axis="y", fixed_value=0.0,
                 alpha=plane_alpha_xz, norm_obj=norm_obj, cmap=cmap)
    _quad_slices(ax, yv, zv_yz, A_yz, fixed_axis="x", fixed_value=0.0,
                 alpha=plane_alpha_yz, norm_obj=norm_obj, cmap=cmap)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    if xlim is not None:
        ax.set_xlim3d(*xlim)
    if ylim is not None:
        ax.set_ylim3d(*ylim)
    if zlim is not None:
        ax.set_zlim3d(*zlim)
    else:
        ax.set_zlim3d(*z_lim_mesh)

    # sensible x/y limits from mesh
    xyz_min, xyz_max = pts.min(axis=0), pts.max(axis=0)
    max_range = (xyz_max - xyz_min).max()
    mid = 0.5 * (xyz_max + xyz_min)

    if xlim is None:
        ax.set_xlim3d(mid[0] - max_range/2, mid[0] + max_range/2)
    if ylim is None:
        ax.set_ylim3d(mid[1] - max_range/2, mid[1] + max_range/2)

    mappable = plt.cm.ScalarMappable(norm=norm_obj, cmap=plt.get_cmap(cmap))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.05)
    cbar.set_label("|p| [Pa]")

    ax.grid(False)
    plt.tight_layout()
    plt.show()


# =========================================================
# Visualization: 2D maps (XZ / YZ)
# =========================================================
title_tag = os.path.basename(CR_FILE)

# XZ
fig1, ax1 = plt.subplots(figsize=(7, 5))
pc1 = ax1.pcolormesh(
    xx, zz,
    20.0*np.log10(np.abs(p_total.reshape(xx.shape)) + 1e-12),
    shading="auto", cmap="bwr", vmin=-50, vmax=50
)
fig1.colorbar(pc1, ax=ax1, label="SPL [dB]")
ax1.set_title(f"Total Sound Field [Loaded Cr: {title_tag}] (XZ)")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("z [m]")
ax1.set_aspect("equal")

fig2, ax2 = plt.subplots(figsize=(7, 5))
pc2 = ax2.pcolormesh(
    xx, zz,
    np.real(p_total.reshape(xx.shape)),
    shading="auto", cmap="bwr"
)
fig2.colorbar(pc2, ax=ax2, label="Re[p]")
ax2.set_title("Reconstructed Sound Field (Real part, XZ)")
ax2.set_xlabel("x [m]")
ax2.set_ylabel("z [m]")
ax2.set_aspect("equal")

fig3, ax3 = plt.subplots(figsize=(7, 5))
pc3 = ax3.pcolormesh(
    xx, zz,
    np.real(p_ref.reshape(xx.shape)),
    shading="auto", cmap="bwr", vmin=-0.15, vmax=0.15
)
fig3.colorbar(pc3, ax=ax3, label="Re[p_reflected]")
ax3.set_title("Reflected Field Only (Real part, XZ)")
ax3.set_xlabel("x [m]")
ax3.set_ylabel("z [m]")
ax3.set_aspect("equal")

fig4, ax4 = plt.subplots(figsize=(7, 5))
pc4 = ax4.pcolormesh(
    xx, zz,
    np.abs(p_total.reshape(xx.shape)),
    shading="auto", cmap=POSTPROC_CMAP,
    vmin=POSTPROC_CLIM[0], vmax=POSTPROC_CLIM[1]
)
fig4.colorbar(pc4, ax=ax4, label="|p_total| [Pa]")
ax4.set_title(f"Total Sound Field Magnitude [Loaded Cr: {title_tag}] (XZ)")
ax4.set_xlabel("x [m]")
ax4.set_ylabel("z [m]")
ax4.set_aspect("equal")

# YZ
fig1y, ax1y = plt.subplots(figsize=(7, 5))
pc1y = ax1y.pcolormesh(
    yy, zz_yz,
    20.0*np.log10(np.abs(p_total_yz.reshape(yy.shape)) + 1e-12),
    shading="auto", cmap="bwr", vmin=-50, vmax=50
)
fig1y.colorbar(pc1y, ax=ax1y, label="SPL [dB]")
ax1y.set_title(f"Total Sound Field [YZ @ x=0]  [Loaded Cr: {title_tag}]")
ax1y.set_xlabel("y [m]")
ax1y.set_ylabel("z [m]")
ax1y.set_aspect("equal")

fig2y, ax2y = plt.subplots(figsize=(7, 5))
pc2y = ax2y.pcolormesh(
    yy, zz_yz,
    np.real(p_total_yz.reshape(yy.shape)),
    shading="auto", cmap="bwr"
)
fig2y.colorbar(pc2y, ax=ax2y, label="Re[p]")
ax2y.set_title("Reconstructed Sound Field (Real part) [YZ @ x=0]")
ax2y.set_xlabel("y [m]")
ax2y.set_ylabel("z [m]")
ax2y.set_aspect("equal")

fig3y, ax3y = plt.subplots(figsize=(7, 5))
pc3y = ax3y.pcolormesh(
    yy, zz_yz,
    np.real(p_ref_yz.reshape(yy.shape)),
    shading="auto", cmap="bwr", vmin=-0.15, vmax=0.15
)
fig3y.colorbar(pc3y, ax=ax3y, label="Re[p_reflected]")
ax3y.set_title("Reflected Field Only (Real part) [YZ @ x=0]")
ax3y.set_xlabel("y [m]")
ax3y.set_ylabel("z [m]")
ax3y.set_aspect("equal")

fig4y, ax4y = plt.subplots(figsize=(7, 5))
pc4y = ax4y.pcolormesh(
    yy, zz_yz,
    np.abs(p_total_yz.reshape(yy.shape)),
    shading="auto", cmap=POSTPROC_CMAP,
    vmin=POSTPROC_CLIM[0], vmax=POSTPROC_CLIM[1]
)
fig4y.colorbar(pc4y, ax=ax4y, label="|p_total| [Pa]")
ax4y.set_title(f"Total Sound Field Magnitude [YZ @ x=0]  [Loaded Cr: {title_tag}]")
ax4y.set_xlabel("y [m]")
ax4y.set_ylabel("z [m]")
ax4y.set_aspect("equal")

# Boundary diagnostics
plt.figure()
plt.title("|p_boundary| / Re / Im")
plt.plot(np.abs(p_b), label="|p|")
plt.plot(np.real(p_b), label="Re")
plt.plot(np.imag(p_b), label="Im")
plt.legend()
plt.grid(alpha=0.25)

plt.figure()
plt.title("|v_boundary| / Re / Im")
plt.plot(np.abs(v_b), label="|v|")
plt.plot(np.real(v_b), label="Re")
plt.plot(np.imag(v_b), label="Im")
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

# ---------- 3D cross-slices ----------
plot_cross_slices_from_2d_grids(
    xv, zv, p_total.reshape(zz.shape),
    yv, zv_yz, p_total_yz.reshape(zz_yz.shape),
    title="3D Cross-slices (|p_total|)",
    alpha_xz=PLANE_ALPHA_XZ,
    alpha_yz=PLANE_ALPHA_YZ,
    cmap=POSTPROC_CMAP,
    clim=POSTPROC_CLIM,
    xlim=POSTPROC_XLIM,
    ylim=POSTPROC_YLIM,
    zlim=POSTPROC_ZLIM
)

# ---------- Mesh + cross-slices (separate figure) ----------
if os.path.exists(MESH_PATH):
    plot_mesh_with_cross_slices(
        mesh_path=MESH_PATH,
        xv=xv, zv=zv, p_xz=p_total.reshape(zz.shape),
        yv=yv, zv_yz=zv_yz, p_yz=p_total_yz.reshape(zz_yz.shape),
        title=f"Mesh + 3D Cross-slices (|p_total|, {os.path.basename(CR_FILE)})",
        z_lim_mesh=Z_LIM_MESH,
        mesh_alpha=0.25,
        plane_alpha_xz=PLANE_ALPHA_XZ,
        plane_alpha_yz=PLANE_ALPHA_YZ,
        cmap=POSTPROC_CMAP,
        clim=POSTPROC_CLIM,
        # optional fixed ranges:
        # xlim=POSTPROC_XLIM, ylim=POSTPROC_YLIM, zlim=POSTPROC_ZLIM,
    )
else:
    print(f"[WARN] MESH_PATH not found (skip overlay figure): {MESH_PATH}")


# =========================================================
# Export for post-processing
# =========================================================
ensure_dir(EXPORT_DIR)

base = EXPORT_TAG or os.path.splitext(os.path.basename(CR_FILE))[0]
export_path = os.path.join(EXPORT_DIR, f"postproc_{base}.npz")

np.savez_compressed(
    export_path,
    # grids
    xv=xv, zv=zv, yv=yv, zv_yz=zv_yz,
    xx=xx, zz=zz, yy=yy, zz_yz=zz_yz,
    # complex fields on slices (xy-order: (len(z), len(x)) etc.)
    p_total_xz=p_total.reshape(zz.shape),
    p_ref_xz=p_ref.reshape(zz.shape),
    p_dir_xz=p_dir.reshape(zz.shape),
    p_total_yz=p_total_yz.reshape(zz_yz.shape),
    p_ref_yz=p_ref_yz.reshape(zz_yz.shape),
    p_dir_yz=p_dir_yz.reshape(zz_yz.shape),
    # boundary
    centers=centers,
    areas=areas,
    normals=normals,
    p_boundary=p_b,
    v_boundary=v_b,
    # k-space / operators (for debugging / reproducibility)
    CR_FILE=os.path.basename(CR_FILE),
    MESH_PATH=np.array(MESH_PATH),
    nk=np.array(nk, dtype=np.int32),
    k=np.array(k, dtype=float),
    f=np.array(f, dtype=float),
    rho=np.array(rho, dtype=float),
    c=np.array(c, dtype=float),
    Z0=np.array(Z0, dtype=float),
    kx_vec=kx,
    ky_vec=ky,
    kz_vec=kz,
    wk=wk,
    # visualization meta
    title_tag=np.array(title_tag),
    cmap=np.array(POSTPROC_CMAP),
    clim=np.array(POSTPROC_CLIM, dtype=float),
    xlim=np.array(POSTPROC_XLIM, dtype=float),
    ylim=np.array(POSTPROC_YLIM, dtype=float),
    zlim=np.array(POSTPROC_ZLIM, dtype=float),
    z_lim_mesh=np.array(Z_LIM_MESH, dtype=float),
    plane_alpha_xz=np.array(PLANE_ALPHA_XZ, dtype=float),
    plane_alpha_yz=np.array(PLANE_ALPHA_YZ, dtype=float),
    # source
    x_src=x_src,
)
print(f"[EXPORT] Saved post-processing data -> {export_path}")

plt.show()
