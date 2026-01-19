#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0_generate_mesh.py

Surface mesh generator for BEM simulations (PoMA reference).

This script generates a 2D surface mesh from a CAD geometry using Gmsh.
The resulting mesh is used for Boundary Element Method (BEM) simulations
in the wavenumber-domain reflection coefficient estimation workflow.

Directory structure (expected):
    geometry/
        slit.step
        flatplate.step
    mesh/

Example:
    python 0_generate_mesh.py
    python 0_generate_mesh.py --geom slit.step

Notes:
- Mesh size is specified in the same unit as the CAD file (typically mm).
- Only surface meshes (2D) are generated.
- Output mesh is written to the mesh/ directory.
"""

import argparse
import os
import gmsh


# =========================================================
# Default settings
# =========================================================
GEOMETRY_DIR = "geometry"
MESH_DIR     = "mesh"

# Geometry candidates (keep both to indicate available cases)
# DEFAULT_GEOM = "flatplate.step"
DEFAULT_GEOM = "slit.step"

# Characteristic length (mesh size in geometry units)
LC_MIN = 25.0
LC_MAX = 30.0


# =========================================================
# Main routine
# =========================================================
def generate_mesh(geom_name: str) -> None:
    """
    Generate a surface mesh from a CAD geometry using Gmsh.

    Parameters
    ----------
    geom_name : str
        Geometry file name located in geometry/
    """

    geom_path = os.path.join(GEOMETRY_DIR, geom_name)
    if not os.path.exists(geom_path):
        raise FileNotFoundError(f"Geometry file not found: {geom_path}")

    os.makedirs(MESH_DIR, exist_ok=True)
    out_path = os.path.join(MESH_DIR, geom_name + ".msh")

    gmsh.initialize()
    try:
        gmsh.open(geom_path)

        # Global mesh size control
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", LC_MIN)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", LC_MAX)

        # Generate surface mesh (2D)
        gmsh.model.mesh.generate(2)

        # Write mesh
        gmsh.write(out_path)

        print("[OK] Surface mesh generated")
        print(f"     Geometry : {geom_path}")
        print(f"     Mesh     : {out_path}")
        print(f"     Mesh size: {LC_MIN} - {LC_MAX} (geometry units)")

    finally:
        gmsh.finalize()


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate a surface mesh for BEM simulations using Gmsh"
    )
    parser.add_argument(
        "--geom",
        type=str,
        default=DEFAULT_GEOM,
        help="Geometry file name in geometry/ (default: slit.step)"
    )

    args = parser.parse_args()
    generate_mesh(args.geom)
