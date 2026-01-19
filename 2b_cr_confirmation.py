# -*- coding: utf-8 -*-
"""
Incident & Reflected field angular maps (Pr only).

- φ = 0° fixed
- θ = 0°, 30°, 60°
- Reflected spectrum Pr is visualized in angular (φ, θ) domain
- Colorbar range:
    plane : [0, 1]
    slit  : [0, 0.5]
- Incident direction is overlaid by an "X" marker (ideal angle)
- Output filenames explicitly include "plane" / "slit"

This script is intended for figure generation directly used in the paper.
"""

# =========================================================
# CONFIG
# =========================================================
# --- Input ---
IN_NPZ = "Cr_from_space_estimated__space_snapshot_f3400Hz_Nrec400_Nsrc100_._mesh_slit.step.msh__nk2401.npz"
# IN_NPZ = "Cr_from_space_estimated__space_snapshot_f3400Hz_Nrec400_Nsrc100_._mesh_flatplate.step.msh__nk2401.npz"

# --- Incident angles ---
PHI_IN_DEG = 0.0
THETA_LIST = [0.0, 30.0, 60.0]

# --- Output ---
PLOT_DIR = "figs_cr_from_npz__PiPr_fixed01"
SHOW_FIG = True
CMAP = "Reds"

# --- Display convention ---
INC_SIGN_FOR_AZIMUTH = +1.0   # incident side (kz < 0)
REF_SIGN_FOR_AZIMUTH = -1.0   # reflected side (kz > 0)

# --- Incident marker (X) ---
SHOW_INCIDENT_X = True
X_MARK_SIZE = 90
X_MARK_LW   = 1.8
X_MARK_Z    = 5

# =========================================================
# Imports
# =========================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# =========================================================
# Utility helpers
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def wrap_phi_deg(phi_deg: np.ndarray) -> np.ndarray:
    """Wrap azimuth angle to [-180, 180) degrees."""
    return (phi_deg + 180.0) % 360.0 - 180.0

def phi_display_from_sign(kx, ky, sign_kz: float):
    """
    Azimuth display rule:
    - sign_kz = +1 : incident-side convention
    - sign_kz = -1 : reflected-side convention
    """
    return wrap_phi_deg(np.degrees(np.arctan2(sign_kz * ky, sign_kz * kx)))

def overlay_incident_marker(ax, phi_deg: float, theta_deg: float):
    """Overlay ideal incident direction as an X marker."""
    ax.scatter(
        np.radians([phi_deg]),
        [theta_deg],
        marker="x",
        s=X_MARK_SIZE,
        linewidths=X_MARK_LW,
        zorder=X_MARK_Z,
    )

def nearest_k_index(kx, ky, k, theta_deg, phi_deg) -> int:
    """Find nearest k-grid index for a given (θ, φ)."""
    th = np.radians(theta_deg)
    ph = np.radians(phi_deg)
    kx_t = k * np.sin(th) * np.cos(ph)
    ky_t = k * np.sin(th) * np.sin(ph)
    return int(np.argmin((kx - kx_t)**2 + (ky - ky_t)**2))

# =========================================================
# Spectral helpers
# =========================================================
def reflected_raw(Cr, e_in):
    return Cr @ e_in

def reflected_weighted(Cr, e_in, wk):
    eps = 1e-300
    sqrtw = np.sqrt(wk + eps)
    return sqrtw * (Cr @ (e_in / sqrtw))

def mmm_weight_from_kz(k: float, kz: np.ndarray, nk: int):
    """MMM hemisphere-to-plane Jacobian weight."""
    return (2.0 * np.pi * k * kz.astype(float)) / float(nk)

# =========================================================
# Plot helpers
# =========================================================
def polar_scatter_pr(
    phi_deg, theta_deg, values,
    fname, outdir,
    cb_min, cb_max,
    cmap,
    xmark=None,
    show=True,
):
    """Polar scatter plot of |Pr| with fixed colorbar range."""
    norm = Normalize(vmin=cb_min, vmax=cb_max)
    vals = np.clip(np.abs(values), cb_min, cb_max)

    fig = plt.figure(figsize=(5.8, 5.2))
    ax = fig.add_subplot(projection="polar")

    sc = ax.scatter(
        np.radians(phi_deg),
        theta_deg,
        c=vals,
        s=26,
        cmap=cmap,
        norm=norm,
    )
    fig.colorbar(sc, ax=ax, label="|Pr|")

    if xmark is not None:
        overlay_incident_marker(ax, xmark[0], xmark[1])

    ax.set_rmin(0)
    ax.set_rmax(90)
    ax.set_rticks([10, 30, 60, 90])

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=160)
    if show:
        plt.show()
    plt.close(fig)

# =========================================================
# Main
# =========================================================
def main():
    ensure_dir(PLOT_DIR)

    # ---------- Load NPZ ----------
    dat = np.load(IN_NPZ, allow_pickle=True)
    Cr = dat["Cr"]
    kx, ky, kz = dat["kx_vec"], dat["ky_vec"], dat["kz_vec"]
    k = float(dat["k"])
    f = float(dat["f"])
    nk = len(kx)

    if "wk" in dat.files:
        wk = np.asarray(dat["wk"], float)
    else:
        wk = mmm_weight_from_kz(k, kz, nk)

    # ---------- Case-dependent colorbar ----------
    name = IN_NPZ.lower()
    if "slit" in name:
        cb_min, cb_max, case_tag = 0.0, 0.5, "slit"
    elif "plane" in name:
        cb_min, cb_max, case_tag = 0.0, 1.0, "plane"
    else:
        cb_min, cb_max, case_tag = 0.0, 1.0, "case"

    # ---------- Angular coordinates ----------
    kr = np.sqrt(kx**2 + ky**2)
    theta_abs_deg = np.degrees(np.arctan2(kr, np.maximum(1e-300, np.abs(kz))))
    phi_ref_deg = phi_display_from_sign(kx, ky, REF_SIGN_FOR_AZIMUTH)

    # ---------- Loop over incident angles ----------
    for theta_in in THETA_LIST:
        idx_in = nearest_k_index(kx, ky, k, theta_in, PHI_IN_DEG)
        e_in = np.zeros(nk, dtype=np.complex128)
        e_in[idx_in] = 1.0

        # Ideal incident marker
        if SHOW_INCIDENT_X:
            phi_x = phi_display_from_sign(
                k * np.sin(np.radians(theta_in)),
                0.0,
                INC_SIGN_FOR_AZIMUTH,
            )
            xmark = (phi_x, theta_in)
        else:
            xmark = None

        # Reflected spectra
        Pr_raw = reflected_raw(Cr, e_in)
        Pr_w   = reflected_weighted(Cr, e_in, wk)

        polar_scatter_pr(
            phi_ref_deg, theta_abs_deg, Pr_raw,
            fname=f"Pr_raw_{case_tag}_theta{int(theta_in):02d}.png",
            outdir=PLOT_DIR,
            cb_min=cb_min, cb_max=cb_max,
            cmap=CMAP,
            xmark=xmark,
            show=SHOW_FIG,
        )

        polar_scatter_pr(
            phi_ref_deg, theta_abs_deg, Pr_w,
            fname=f"Pr_w_{case_tag}_theta{int(theta_in):02d}.png",
            outdir=PLOT_DIR,
            cb_min=cb_min, cb_max=cb_max,
            cmap=CMAP,
            xmark=xmark,
            show=SHOW_FIG,
        )

    print(f"[DONE] Saved figures to: {os.path.abspath(PLOT_DIR)}")
    print(f"Case = '{case_tag}', Colorbar range = [{cb_min}, {cb_max}]")
    print(f"Input file = {IN_NPZ}")

# =========================================================
if __name__ == "__main__":
    main()
