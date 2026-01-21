# BEM Cr Study (PoMA 2025)

This repository contains the full numerical pipeline used in the PoMA paper presented at the **ASA\UTF{2013}ASJ Joint Meeting 2025 (Honolulu)**.

## Related Publications

* **PoMA (Proceedings of Meetings on Acoustics)**
  ASA\UTF{2013}ASJ Joint Meeting 2025, Honolulu
  Official conference site: [https://acousticalsociety.org/honolulu-2025/](https://acousticalsociety.org/honolulu-2025/)

* **arXiv preprint**
  *Directional Reflection Modeling via Wavenumber-Domain Reflection Coefficient for 3D Acoustic Field Simulation*
  [https://arxiv.org/abs/2601.07481](https://arxiv.org/abs/2601.07481)

## Purpose

The goal of this repository is to validate and demonstrate a **wavenumber-domain acoustic reflection coefficient (Cr)** approach for 3D sound field simulation using the Boundary Element Method (BEM).

The method enables:

* Direction-dependent reflection modeling
* Nonlocal boundary operators derived from measured or simulated sound fields
* Quantitative comparison with conventional (scalar-admittance) BEM models

## Repository Structure

```
.
├── 0_generate_mesh.py
├── 1_bem_export_space_fibonacci.py
├── 1b_layout_plot_fibonacci.py
├── 2_estimate_cr_from_space.py
├── 2b_cr_confirmation.py
├── 3_bem_forward_proposed.py
├── 4_bem_forward_legacyY.py
├── 5_postproc_view.py
├── geometry/
│   ├── flatplate.step
│   └── slit.step
├── mesh/
│   ├── flatplate.step.msh
│   └── slit.step.msh
├── postproc_data/
├── figs_cr_from_npz__PiPr_fixed01/
└── README.md
```

## Pipeline Overview

```
[ Geometry / Mesh ]
        |
        v
(0) Mesh generation
        |
        v
(1) BEM: space snapshot (Fibonacci sources)
        |
        v
(2) Estimate Cr in wavenumber domain
        |
        v
(3) Forward BEM with Cr-based admittance (proposed)
        |
        +--> (4) Forward BEM with scalar admittance (legacy)
        |
        v
(5) Post-processing & comparison
      - 2D sections (XZ / YZ)
      - 3D cross-slices
      - Mesh overlay
      - Cosine similarity / MSE
```

## Key Scripts

* **3_bem_forward_proposed.py**
  Forward BEM using nonlocal, k-space admittance derived from Cr.

* **4_bem_forward_legacyY.py**
  Baseline BEM using scalar (local) admittance Y.

* **5_postproc_view.py**
  Visualization and quantitative comparison tool:

  * 2D slices
  * 3D orthogonal cross-slices
  * Mesh overlay
  * Cosine similarity & MSE (with / without absolute value)

## Metrics

* **Cosine similarity**
  Evaluates shape similarity of sound fields

* **Mean Squared Error (MSE)**
  Evaluates absolute magnitude difference

Both metrics are evaluated for:

* XZ plane
* YZ plane
* Combined (XZ + YZ)

## Notes

* This repository is intended for **research and reproducibility**.
* The code prioritizes clarity and physical consistency over raw performance.
* Meshes are assumed to be in **mm** unless otherwise specified.

## Author

Satoshi Hoshika
Graduate School of Design, Kyushu University
(ASA\UTF{2013}ASJ Joint Meeting 2025, PoMA)
