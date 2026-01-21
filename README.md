BEM Cr Study (PoMA 2025)
======================

Wavenumber-domain reflection coefficient (Cr) based Boundary Element Method (BEM) study
for directional reflection modeling and nonlocal boundary operators.

This repository contains the full numerical pipeline used in the PoMA 2025 submission,
from geometry / mesh generation to Cr estimation, forward BEM simulation, and post-processing.


Overview
--------

Purpose
- Estimate wavenumber-domain reflection coefficient (Cr) from spatial sound field data
- Construct nonlocal boundary operators using Cr
- Compare:
  * Proposed Cr-based BEM
  * Legacy scalar-admittance BEM
- Visualize and quantitatively compare results
  (Cosine similarity, MSE, abs / complex)


Pipeline
--------

[ geometry/*.step ]
        |
        v
[0_generate_mesh.py]
        |
        v
[ mesh/*.msh ]
        |
        v
[1_bem_export_space_fibonacci.py]
        |   (space snapshot: Pi / Pr)
        v
[ space_snapshot_*.npz ]
        |
        v
[2_estimate_cr_from_space.py]
        |
        v
[ Cr_from_space_estimated_*.npz ]
        |
        |-- 2b_cr_confirmation.py
        |       (Cr visualization & sanity check)
        |
        v
[3_bem_forward_proposed.py]
        |   (Cr-based nonlocal BEM)
        v
[ postproc_*.npz ]
        |
        |-- 4_bem_forward_legacyY.py
        |       (scalar admittance baseline)
        |
        v
[5_postproc_view.py]
        |
        |-- 2D sections (XZ / YZ)
        |-- 3D cross-slices
        |-- Mesh + slice overlay
        |-- Quantitative comparison
            (CosSim / MSE, abs & complex)


Repository Structure
--------------------

.
|-- geometry/
|   |-- flatplate.step
|   `-- slit.step
|
|-- mesh/
|   |-- flatplate.step.msh
|   `-- slit.step.msh
|
|-- postproc_data/
|   `-- postproc_*.npz
|
|-- figs_cr_from_npz__PiPr_fixed01/
|   `-- (Cr visualization figures)
|
|-- 0_generate_mesh.py
|-- 1_bem_export_space_fibonacci.py
|-- 1b_layout_plot_fibonacci.py
|-- 2_estimate_cr_from_space.py
|-- 2b_cr_confirmation.py
|-- 3_bem_forward_proposed.py
|-- 4_bem_forward_legacyY.py
|-- 5_postproc_view.py
|
|-- README.txt
`-- .gitignore


Scripts Description
-------------------

0_generate_mesh.py
- Generate surface mesh from CAD geometry using Gmsh
- Supports flat plate and slit geometries
- Output: mesh/*.msh

1_bem_export_space_fibonacci.py
- Forward BEM simulation for space snapshot acquisition
- Fibonacci sampling for source directions
- Output: space_snapshot_*.npz

2_estimate_cr_from_space.py
- Estimate Cr(kx, ky) from spatial sound field
- Weighted k-space formulation
- Output: Cr_from_space_estimated_*.npz

2b_cr_confirmation.py
- Visualization and sanity check of estimated Cr

3_bem_forward_proposed.py
- Forward BEM using Cr-based nonlocal boundary operator
- Main proposed method

4_bem_forward_legacyY.py
- Baseline BEM with scalar admittance Y

5_postproc_view.py
- Pure post-processing & visualization tool
- 2D sections (XZ / YZ)
- 3D orthogonal cross-slices
- Mesh + slice overlay
- Quantitative comparison (CosSim / MSE, abs & complex)


Typical Usage
-------------

python 0_generate_mesh.py
python 1_bem_export_space_fibonacci.py
python 2_estimate_cr_from_space.py
python 2b_cr_confirmation.py
python 3_bem_forward_proposed.py
python 4_bem_forward_legacyY.py
python 5_postproc_view.py


Notes
-----

- Mesh unit: mm (internally converted to meters)
- Flat plate and slit structures supported
- Solver-independent post-processing
- No commercial solvers required


Citation
--------

S. Hoshika et al.
Directional Reflection Modeling via Wavenumber-Domain Reflection Coefficient for 3D Acoustic Field Simulation
Proceedings of Meetings on Acoustics (PoMA), 2025.


Author
------

Satoshi Hoshika
Graduate School of Design, Kyushu University