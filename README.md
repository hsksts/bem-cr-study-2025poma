# BEM Cr Study (PoMA 2025)

This repository contains the **full numerical pipeline** used in the PoMA 2025 study
on **wavenumber-domain reflection coefficients (Cr)** and their validation via BEM.

The filenames below are **authoritative** and match the actual programs in this repository.

---

## Directory Structure

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
│
├── geometry/
│   ├── flatplate.step
│   └── slit.step
│
├── mesh/
│   ├── flatplate.step.msh
│   └── slit.step.msh
│
├── postproc_data/
├── figs_cr_from_npz__PiPr_fixed01/
└── README.md
```

---

## Pipeline Overview (ASCII)

```
(0) Geometry & Mesh
    └─ 0_generate_mesh.py
         ↓
(1) BEM field sampling (space snapshot)
    └─ 1_bem_export_space_fibonacci.py
         ↓
    (layout check)
    └─ 1b_layout_plot_fibonacci.py
         ↓
(2) Cr estimation (k-space)
    └─ 2_estimate_cr_from_space.py
         ↓
    (diagnostics / confirmation)
    └─ 2b_cr_confirmation.py
         ↓
(3) Forward BEM (proposed, k-space admittance)
    └─ 3_bem_forward_proposed.py
         ↓
(4) Forward BEM (baseline, scalar Y)
    └─ 4_bem_forward_legacyY.py
         ↓
(5) Visualization & quantitative comparison
    └─ 5_postproc_view.py
```

---

## Script Descriptions

### 0_generate_mesh.py
- Generates surface geometry and Gmsh meshes
- Supports **flat plate** and **slit structure**
- Outputs `.step` and `.msh`

### 1_bem_export_space_fibonacci.py
- Boundary Element Method (BEM)
- Samples sound field using **Fibonacci-distributed sources**
- Exports spatial pressure snapshots (`.npz`)

### 1b_layout_plot_fibonacci.py
- Visual sanity check of source / receiver layout

### 2_estimate_cr_from_space.py
- Estimates **wavenumber-domain reflection coefficient Cr**
- Uses spatial snapshots
- Outputs `Cr_*.npz`

### 2b_cr_confirmation.py
- Visualizes Pi / Pr / Cr
- Angle-dependent reflection diagnostics

### 3_bem_forward_proposed.py
- Forward BEM using **k-space admittance operator**
- Loads estimated Cr directly
- Produces validation sound fields

### 4_bem_forward_legacyY.py
- Baseline BEM with **scalar admittance Y**
- Used as reference for comparison

### 5_postproc_view.py
- Pure post-processing & visualization
- 2D sections (XZ / YZ)
- 3D orthogonal cross-slices
- Mesh + slice overlay
- **Quantitative comparison**
  - Cosine similarity
  - MSE
  - abs / complex-field options

---

## Notes

- No acoustic computation is performed in `5_postproc_view.py`
- Mesh units are assumed **mm → m** unless otherwise stated
- Designed for **PoMA / JASA-level reproducibility**
- Flat plate and slit cases are switchable via file paths

---

## Author

Satoshi Hoshika  
Graduate School of Design, Kyushu University
