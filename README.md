# Wavenumber-Domain Reflection Coefficient Estimation (PoMA Reference Code)

This repository provides **research-oriented reference implementations**
used for numerical validation in a PoMA paper on **wavenumber-domain acoustic
reflection coefficient estimation** and its application to boundary element
methods (BEM).

The codes prioritize **clarity, reproducibility, and numerical transparency**
over computational efficiency or large-scale applicability.

---

## Scope and purpose

This repository is intended to:

- Demonstrate the full numerical workflow  
  **(sound field generation → wavenumber-domain estimation → BEM reloading)**.
- Provide reproducible reference results for **methodological comparison**.
- Support **quantitative validation** against conventional admittance-based
  boundary conditions.

The implementations are **not optimized solvers** and should not be interpreted
as production-ready BEM/FEM software.

---

## Repository structure

```
.
├── geometry/
│   └── plane.step
│
├── mesh/
│   └── generate_mesh.py
│
├── bem/
│   └── bem_export_space_fibonacci.py
│
├── estimation/
│   └── estimate_cr_from_space.py
│
├── bem_reload/
│   ├── bem_reload_standard_ft.py
│   └── bem_legacyY_scalar.py
│
├── postproc/
│   └── postproc_view.py
│
└── README.md
```

---

## Numerical workflow

### Step 1: Mesh generation
```bash
python mesh/generate_mesh.py
```

### Step 2: BEM sound field generation (space domain)
```bash
python bem/bem_export_space_fibonacci.py
```

### Step 3: Wavenumber-domain reflection coefficient estimation
```bash
python estimation/estimate_cr_from_space.py
```

### Step 4: BEM reloading with estimated boundary operator

**Standard FT-based boundary operator**
```bash
python bem_reload/bem_reload_standard_ft.py
```

**Conventional scalar admittance (baseline)**
```bash
python bem_reload/bem_legacyY_scalar.py
```

### Step 5: Post-processing and comparison
```bash
python postproc/postproc_view.py
```

---

## Quantitative comparison metrics

- Cosine similarity
- Mean squared error (MSE)

Evaluated on:
- XZ cross-section
- YZ cross-section
- Combined XZ + YZ field

---

## Notes and limitations

- Reference implementation for PoMA validation
- Small-to-medium mesh sizes assumed
- Cr stored without quadrature weights

---

## Reproducibility

All scripts are deterministic.
Results can be reproduced by running the scripts in the listed order.

---

## License

Academic and research use only.
Please cite the associated PoMA paper when using this code.

## Citation

S. Hoshika et al.
Directional Reflection Modeling via Wavenumber-Domain Reflection Coefficient for 3D Acoustic Field Simulation
Proceedings of Meetings on Acoustics (PoMA), 2025.


## Author
Satoshi Hoshika
Graduate School of Design, Kyushu University
