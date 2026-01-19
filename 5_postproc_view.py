#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing viewer for BEM slice data (publication-ready, FULL VERSION).

Capabilities
------------
(A) 2D cross-sections (XZ / YZ)
(B) 3D orthogonal cross-slices (XZ + YZ)
(C) Mesh + cross-slice overlay
(D) Quantitative comparison (Cosine similarity, MSE)

This script performs NO acoustic computation.
It only visualizes and compares NPZ outputs.

Author
------
satoshihoshika
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import meshio


# =========================================================
# Matplotlib style (paper-friendly)
# =========================================================
matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.size": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.labelpad": 4,
    "axes.titlepad": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================================================
# Utilities
# =========================================================
def to_ij_abs(A, X, Z):
    """Convert A to ij ordering (len(X), len(Z)) as abs values."""
    if A.shape == (len(Z), len(X)):   # meshgrid(indexing="xy") output
        return np.abs(A).T
    if A.shape == (len(X), len(Z)):
        return np.abs(A)
    raise ValueError(f"Array shape incompatible: A{A.shape}, X{len(X)}, Z{len(Z)}")


def quad_slices(ax, X1, X2, A, fixed_axis, fixed_value,
                alpha=1.0, norm=None, cmap="Reds"):
    """Plot a colored plane by splitting into 4 quads (depth-sort safe)."""
    n1, n2 = A.shape
    i1, i2 = n1 // 2, n2 // 2
    blocks = [
        (slice(0, i1), slice(0, i2)),
        (slice(0, i1), slice(i2, n2)),
        (slice(i1, n1), slice(0, i2)),
        (slice(i1, n1), slice(i2, n2)),
    ]

    cmap_obj = plt.get_cmap(cmap)

    for s1, s2 in blocks:
        G1, G2 = np.meshgrid(X1[s1], X2[s2], indexing="ij")
        colors = cmap_obj(norm(A[s1, s2])) if norm is not None else None

        if fixed_axis == "y":
            X, Y, Z = G1, np.full_like(G1, fixed_value), G2
        elif fixed_axis == "x":
            X, Y, Z = np.full_like(G1, fixed_value), G1, G2
        else:
            raise ValueError("fixed_axis must be 'x' or 'y'")

        ax.plot_surface(
            X, Y, Z,
            facecolors=colors,
            rstride=1, cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=alpha,
        )


def set_axes_from_bounds(ax, xyz_min, xyz_max, pad=0.02):
    """
    Set 3D axes limits from bounding box with equal aspect.
    pad: fraction of max span.
    """
    xyz_min = np.asarray(xyz_min, float)
    xyz_max = np.asarray(xyz_max, float)

    span = xyz_max - xyz_min
    L = float(np.max(span))
    L = max(L, 1e-12)
    p = pad * L

    mid = 0.5 * (xyz_min + xyz_max)

    ax.set_xlim(mid[0] - 0.5 * L - p, mid[0] + 0.5 * L + p)
    ax.set_ylim(mid[1] - 0.5 * L - p, mid[1] + 0.5 * L + p)
    ax.set_zlim(mid[2] - 0.5 * L - p, mid[2] + 0.5 * L + p)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def mesh_faces_from_meshio(mesh, pts):
    """
    Extract surface faces as list of (nverts,3) arrays for Poly3DCollection.
    Supports triangles/quads. If only "line" exists, returns empty.
    """
    faces = []
    for cell in mesh.cells:
        if cell.type == "triangle":
            tri = cell.data.astype(np.int64)
            faces.extend(pts[tri])  # each is (3,3)
        elif cell.type == "quad":
            q = cell.data.astype(np.int64)
            faces.extend(pts[q])    # each is (4,3)
    return faces


def mesh_edges_from_meshio(mesh, pts):
    """
    Extract edges (line cells) for wireframe.
    Returns list of (2,3) arrays.
    """
    edges = []
    for cell in mesh.cells:
        if cell.type == "line":
            e = cell.data.astype(np.int64)
            edges.extend(pts[e])  # each is (2,3)
    return edges


# =========================================================
# Quantitative comparison (MSE / CosSim)
# =========================================================
def compare_npz_files(npz1, npz2, use_abs=True):

    d1 = np.load(npz1, allow_pickle=True)
    d2 = np.load(npz2, allow_pickle=True)

    def vec(a):
        return np.abs(a).ravel() if use_abs else a.ravel()

    a_xz, b_xz = vec(d1["p_total_xz"]), vec(d2["p_total_xz"])
    a_yz, b_yz = vec(d1["p_total_yz"]), vec(d2["p_total_yz"])

    def cosine(u, v):
        d = np.linalg.norm(u) * np.linalg.norm(v)
        return np.nan if d < 1e-12 else float(np.dot(u.conjugate(), v).real / d)

    cos_xz = cosine(a_xz, b_xz)
    mse_xz = float(np.mean((a_xz - b_xz) ** 2))

    cos_yz = cosine(a_yz, b_yz)
    mse_yz = float(np.mean((a_yz - b_yz) ** 2))

    a_all = np.concatenate([a_xz, a_yz])
    b_all = np.concatenate([b_xz, b_yz])

    cos_all = cosine(a_all, b_all)
    mse_all = float(np.mean((a_all - b_all) ** 2))

    print("\n================ NPZ COMPARISON ================")
    print(f"[XZ]    Cosine = {cos_xz:.6f},  MSE = {mse_xz:.6e}")
    print(f"[YZ]    Cosine = {cos_yz:.6f},  MSE = {mse_yz:.6e}")
    print(f"[TOTAL] Cosine = {cos_all:.6f}, MSE = {mse_all:.6e}")
    print("================================================\n")

    return dict(
        cos_xz=cos_xz, mse_xz=mse_xz,
        cos_yz=cos_yz, mse_yz=mse_yz,
        cos_total=cos_all, mse_total=mse_all
    )


# =========================================================
# 2D sections
# =========================================================
def plot_2d_sections(xv, zv, p_xz, yv, zv_yz, p_yz, clim):

    XX, ZZ = np.meshgrid(xv, zv, indexing="xy")
    YY, ZZy = np.meshgrid(yv, zv_yz, indexing="xy")

    fig, axs = plt.subplots(2, 2, figsize=(6, 5), constrained_layout=True)

    axs[0, 0].pcolormesh(XX, ZZ, 20 * np.log10(np.abs(p_xz) + 1e-12),
                         cmap="bwr", vmin=-50, vmax=50, shading="auto")
    axs[0, 1].pcolormesh(XX, ZZ, np.abs(p_xz),
                         cmap="Reds", vmin=clim[0], vmax=clim[1], shading="auto")
    axs[1, 0].pcolormesh(YY, ZZy, 20 * np.log10(np.abs(p_yz) + 1e-12),
                         cmap="bwr", vmin=-50, vmax=50, shading="auto")
    axs[1, 1].pcolormesh(YY, ZZy, np.abs(p_yz),
                         cmap="Reds", vmin=clim[0], vmax=clim[1], shading="auto")

    for ax in axs.ravel():
        ax.set_aspect("equal")
        ax.set_xlabel("x / y [m]")
        ax.set_ylabel("z [m]")
        ax.grid(False)

    plt.show()


# =========================================================
# 3D cross-slices (no mesh)
# =========================================================
def plot_cross_slices_3d(xv, zv, p_xz, yv, zv_yz, p_yz, clim):

    A_xz = to_ij_abs(p_xz, xv, zv)
    A_yz = to_ij_abs(p_yz, yv, zv_yz)
    norm = Normalize(vmin=clim[0], vmax=clim[1])

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    quad_slices(ax, xv, zv, A_xz, "y", 0.0, alpha=0.95, norm=norm)
    quad_slices(ax, yv, zv_yz, A_yz, "x", 0.0, alpha=0.85, norm=norm)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    # bounds from slices
    xyz_min = np.array([min(xv.min(), 0.0), min(yv.min(), 0.0), min(zv.min(), zv_yz.min())])
    xyz_max = np.array([max(xv.max(), 0.0), max(yv.max(), 0.0), max(zv.max(), zv_yz.max())])
    set_axes_from_bounds(ax, xyz_min, xyz_max)

    plt.show()


# =========================================================
# Mesh + cross-slices overlay (FIXED)
# =========================================================
def plot_mesh_overlay(mesh_path, xv, zv, p_xz, yv, zv_yz, p_yz, clim,
                      mesh_scale=1e-3, mesh_alpha=0.25, edge_alpha=0.35):
    """
    mesh_scale:
      - 1e-3 if mesh is in mm (typical gmsh .step -> .msh)
      - 1.0  if mesh already in meters
    """
    mesh = meshio.read(mesh_path)
    pts = mesh.points[:, :3].astype(float) * float(mesh_scale)

    faces = mesh_faces_from_meshio(mesh, pts)
    edges = mesh_edges_from_meshio(mesh, pts)

    if (not faces) and (not edges):
        print("[WARN] No drawable surface faces/edges found in mesh:", mesh_path)
        return

    A_xz = to_ij_abs(p_xz, xv, zv)
    A_yz = to_ij_abs(p_yz, yv, zv_yz)
    norm = Normalize(vmin=clim[0], vmax=clim[1])

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # --- mesh faces ---
    if faces:
        poly = Poly3DCollection(
            faces,
            facecolors="#d9d9d9",
            edgecolors="none",
            linewidths=0.0,
            alpha=mesh_alpha,
        )
        ax.add_collection3d(poly)

    # --- mesh edges (if present) ---
    if edges:
        for e in edges:
            ax.plot(e[:, 0], e[:, 1], e[:, 2], lw=0.5, color="k", alpha=edge_alpha)

    # --- slices ---
    quad_slices(ax, xv, zv, A_xz, "y", 0.0, alpha=0.95, norm=norm)
    quad_slices(ax, yv, zv_yz, A_yz, "x", 0.0, alpha=0.85, norm=norm)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.grid(False)

    # bounds = merge(mesh bbox and slice bbox
    mesh_min = pts.min(axis=0)
    mesh_max = pts.max(axis=0)

    slice_min = np.array([min(xv.min(), 0.0), min(yv.min(), 0.0), min(zv.min(), zv_yz.min())])
    slice_max = np.array([max(xv.max(), 0.0), max(yv.max(), 0.0), max(zv.max(), zv_yz.max())])

    xyz_min = np.minimum(mesh_min, slice_min)
    xyz_max = np.maximum(mesh_max, slice_max)

    set_axes_from_bounds(ax, xyz_min, xyz_max, pad=0.03)

    plt.show()


# =========================================================
# Main
# =========================================================
def main(npz_path, mesh_path=None, compare_npz=None,
         mesh_scale=1e-3):

    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)

    xv, zv = data["xv"], data["zv"]
    yv, zv_yz = data["yv"], data["zv_yz"]
    p_xz, p_yz = data["p_total_xz"], data["p_total_yz"]
    clim = tuple(data.get("clim", (0.0, 1.0)))

    plot_2d_sections(xv, zv, p_xz, yv, zv_yz, p_yz, clim)
    plot_cross_slices_3d(xv, zv, p_xz, yv, zv_yz, p_yz, clim)

    if mesh_path:
        if os.path.exists(mesh_path):
            plot_mesh_overlay(mesh_path, xv, zv, p_xz, yv, zv_yz, p_yz, clim,
                              mesh_scale=mesh_scale)
        else:
            print("[WARN] mesh_path not found:", mesh_path)

    if compare_npz:
        if os.path.exists(compare_npz):
            compare_npz_files(npz_path, compare_npz)
        else:
            print("[WARN] compare_npz not found:", compare_npz)


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":

    # npz_path = "postproc_data/postproc_Cr_from_space_estimated__space_snapshot_f3400Hz_Nrec400_Nsrc100_._mesh_flatplate.step.msh__nk2401.npz"
    # mesh_path = "./mesh/flatplate.step.msh"
    # compare_npz = "postproc_data/postproc_legacyY__flatplate.step.npz"
    
    npz_path = "postproc_data/postproc_Cr_from_space_estimated__space_snapshot_f3400Hz_Nrec400_Nsrc100_._mesh_slit.step.msh__nk2401.npz"
    mesh_path = "./mesh/slit.step.msh"
    compare_npz = "postproc_data/postproc_legacyY__slit.step.npz"

    # mesh_scale:
    #   1e-3 -> mesh is in mm
    #   1.0  -> mesh is already in m
    main(npz_path, mesh_path, compare_npz, mesh_scale=1e-3)
