#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimate_cr_from_space_static.py

Estimate wavenumber-domain reflection coefficient Cr
from spatial pressure snapshots obtained by BEM.

Pipeline:
  spatial (p_dir, p_ref)
    -> k-space transform
    -> Cr estimation (LS / optional L1)
    -> visualization
    -> export Cr (NPZ, unweighted)

Notes:
- This script is intended for *static post-processing* (no CLI).
- Exported Cr follows the convention:  Pr = Cr @ Pi  (no weights baked in).
- Geometry examples (slit / flatplate) are handled by switching input NPZ.

Author: satoshihoshika
"""

# =========================================================
# CONFIG
# =========================================================

# ---- Input snapshot (from BEM) ----
IN_NPZ = "space_snapshot_f3400Hz_Nrec400_Nsrc100_._mesh_slit.step.msh.npz"
# IN_NPZ = "space_snapshot_f3400Hz_Nrec400_Nsrc100_._mesh_flatplate.step.msh.npz"

# ---- k-space sampling ----
# NK = 400
NK = 2401                          # number of k-space samples (hemisphere Fibonacci)

WINDOW = "rect"                    # "rect" or "hann"

# ---- Estimation options ----
DO_LASSO    = True
LASSO_ALPHA = 5e-5

# ---- Visualization settings ----
THETA_IN_DEG = 30.0                # incident polar angle (deg)
PHI_IN_DEG   = 0.0                 # incident azimuth (deg)

PLOT_DIR = "figs_cr"
SHOW_FIG = True

NPHI   = 200
NTHETA = 200

# ---- Export ----
SAVE_KSPACE         = True
EXPORT_CR_FOR_BEM   = True
CR_EXPORT_FILENAME  = None

# =========================================================
# Imports
# =========================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# =========================================================
# Visualization settings
# =========================================================
CMAP = "Reds"

# φ display sign convention
INC_SIGN_FOR_AZIMUTH = +1.0   # incident: kz < 0
REF_SIGN_FOR_AZIMUTH = -1.0   # reflected: kz > 0


# =========================================================
# Helper functions
# =========================================================

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ---------- Hemisphere Fibonacci sampling ----------
def hemi_fibonacci(nk, k):
    """
    Generate hemisphere (kz >= 0) Fibonacci sampling in k-space.
    """
    i = np.arange(nk) + 0.5
    golden = (1 + np.sqrt(5.0)) / 2.0

    phi = 2*np.pi * (i / golden)
    z   = np.linspace(0.0, 1.0, nk)
    rxy = np.sqrt(np.maximum(1.0 - z**2, 0.0))

    kx = k * rxy * np.cos(phi)
    ky = k * rxy * np.sin(phi)
    kz = k * z

    theta = np.arctan2(rxy, z + 1e-12)
    phi   = np.mod(phi, 2*np.pi)

    return kx, ky, kz, theta, phi


# ---------- Window on receiver grid ----------
def build_window(rec_points, window="rect"):
    x = rec_points[:,0]
    y = rec_points[:,1]

    ux, ix = np.unique(np.round(x, 12), return_inverse=True)
    uy, iy = np.unique(np.round(y, 12), return_inverse=True)
    Nx, Ny = len(ux), len(uy)

    if window == "rect":
        w2d = np.ones((Ny, Nx))
    else:
        wx = 0.5 - 0.5*np.cos(2*np.pi*np.arange(Nx)/(max(Nx-1,1)))
        wy = 0.5 - 0.5*np.cos(2*np.pi*np.arange(Ny)/(max(Ny-1,1)))
        w2d = np.outer(wy, wx)

    w = w2d[iy, ix]
    w *= np.sqrt(len(w) / np.sum(w**2))   # energy normalization

    return w.astype(np.float32)


# ---------- Fourier matrix ----------
def build_F(kx_vec, ky_vec, rec_points):
    x = rec_points[:,0]
    y = rec_points[:,1]
    nk = len(kx_vec)
    return (1/np.sqrt(nk)) * np.exp(-1j * (np.outer(kx_vec, x) + np.outer(ky_vec, y)))


# =========================================================
# Estimators
# =========================================================

def estimate_ls(Pi_k, Pr_k):
    """
    Least-squares estimation of Cr (unweighted).
    """
    s = np.linalg.svd(Pi_k, compute_uv=False)
    cond = s[0] / max(s[-1], 1e-16)
    print(f"[COND] cond(Pi_k) ≈ {cond:.2e}")

    Pi_pinv = np.linalg.pinv(Pi_k)
    Cr_ls = Pr_k @ Pi_pinv

    rel = np.linalg.norm(Cr_ls @ Pi_k - Pr_k) / max(np.linalg.norm(Pr_k), 1e-16)
    return Cr_ls, rel, s


def estimate_l1(Pi_k, Pr_k, alpha=1e-6):
    """
    L1-regularized estimation (row-wise LASSO).
    """
    try:
        from sklearn.linear_model import Lasso
        from joblib import Parallel, delayed
    except Exception as e:
        print(f"[WARN] LASSO unavailable ({e})")
        return None, None, None

    N_out, Ns = Pr_k.shape
    N_in = Pi_k.shape[0]

    Xr, Xi = Pi_k.real.T, Pi_k.imag.T
    X = np.block([[Xr, -Xi], [Xi, Xr]])

    def _fit(i):
        y = np.r_[Pr_k[i].real, Pr_k[i].imag]
        mdl = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000)
        mdl.fit(X, y)
        c = mdl.coef_
        return c[:N_in] + 1j*c[N_in:]

    rows = Parallel(n_jobs=-1)(delayed(_fit)(i) for i in range(N_out))
    Cr = np.stack(rows)

    rel = np.linalg.norm(Cr @ Pi_k - Pr_k) / max(np.linalg.norm(Pr_k),1e-16)
    sparsity = np.mean(np.abs(Cr) < 1e-8)

    return Cr, rel, sparsity


# =========================================================
# Polar visualization helpers
# =========================================================

def _phi_wrap_deg(phi):
    return (phi + 180.0) % 360.0 - 180.0


def phi_disp_from_sign(kx, ky, sign_kz):
    return _phi_wrap_deg(np.degrees(np.arctan2(sign_kz*ky, sign_kz*kx)))


def polar_scatter(phi_deg, theta_deg, vals, title, fname):
    fig = plt.figure(figsize=(5.4,5))
    ax = fig.add_subplot(projection='polar')
    sc = ax.scatter(np.radians(phi_deg), theta_deg,
                    c=np.abs(vals), s=28, cmap=CMAP)
    fig.colorbar(sc, ax=ax, label="|P|")
    ax.set_title(title)
    ax.set_rmax(90)
    ax.set_rticks([10,30,60,90])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=160)
    if SHOW_FIG:
        plt.show()
    plt.close(fig)


# =========================================================
# Main
# =========================================================

def main():

    ensure_dir(PLOT_DIR)

    # ---- Load snapshot ----
    data = np.load(IN_NPZ, allow_pickle=True)
    p_dir = data["p_dir"]
    p_ref = data["p_ref"]
    rec_points = data["rec_points"]
    f = float(data["f"])
    k = float(data["k"])
    z_m = float(data["z_m"])

    Nsrc, Nrec = p_dir.shape
    print(f"[LOAD] {IN_NPZ}")
    print(f"       f={f:.1f} Hz, k={k:.6f}, Nsrc={Nsrc}, Nrec={Nrec}")

    # ---- k-space sampling ----
    nk = int(NK)
    kx, ky, kz, _, _ = hemi_fibonacci(nk, k)

    # ---- Window & transform ----
    w = build_window(rec_points, WINDOW)
    W = np.diag(w)
    F = build_F(kx, ky, rec_points)

    Pi_k = F @ (W @ p_dir.T)
    Pr_k = F @ (W @ p_ref.T)

    kz_eff = np.sqrt(np.maximum(k**2 - kx**2 - ky**2, 0.0))
    Pi_k *= np.exp( 1j * kz_eff * z_m)[:,None]
    Pr_k *= np.exp(-1j * kz_eff * z_m)[:,None]

    # ---- LS estimation ----
    Cr_ls, rel_ls, _ = estimate_ls(Pi_k, Pr_k)
    print(f"[LS] relative error = {rel_ls:.3e}")

    # ---- L1 estimation ----
    Cr_l1 = None
    if DO_LASSO:
        Cr_l1, rel_l1, sp = estimate_l1(Pi_k, Pr_k, LASSO_ALPHA)
        if Cr_l1 is not None:
            print(f"[L1] relative error = {rel_l1:.3e}, sparsity ≈ {sp*100:.1f}%")

    # ---- Export ----
    if EXPORT_CR_FOR_BEM:
        base = os.path.splitext(os.path.basename(IN_NPZ))[0]
        out = f"Cr_from_space_estimated__{base}__nk{nk}.npz"
        Cr_out = Cr_l1 if (Cr_l1 is not None) else Cr_ls

        np.savez_compressed(
            out,
            Cr=Cr_out,
            kx_vec=kx,
            ky_vec=ky,
            kz_vec=kz,
            k=np.array(k),
            f=np.array(f)
        )
        print(f"[OK] Saved Cr: {out}")
        print("Convention: Pr = Cr @ Pi")


if __name__ == "__main__":
    main()
