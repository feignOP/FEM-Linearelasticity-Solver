# 2D Linear Elasticity FEM Solver (Python)

This repository contains a modular finite element framework for solving two dimensional linear elasticity problems. The code is structured to clearly separate FEM utilities, physics kernels, drivers, and post processing.

## Overview

This project provides a compact and extensible FEM codebase that

- Solves 2D plane strain linear elasticity problems
- Uses Q4 elements with Gaussian quadrature
- Supports multiple stress recovery techniques
- Exports VTK files for ParaView visualization
- Demonstrates the full workflow via drivers and a Jupyter notebook

## Repository Structure

```
FemFrameWork/            # Shape functions, Gauss quadrature, FEM utilities
Physics_models/          # Linear elasticity kernels and drivers
Kernel/                  # Assembly, stress recovery, error norms, VTK output
DataFiles/               # Meshes and boundary condition data (.npz)
VTKoutputs/              # Visualization output files
test_functions/          # Verification and test scripts
pictures/                # Figures and mesh visuals
elasticity.ipynb         # Main example notebook
add_paths.py
README.md
```

## Physics Model

- Plane strain linear elasticity
- Constitutive law defined using Lamé parameters
- Gravity and edge shear loading supported
- Stress recovery at Gauss points, element nodes, and averaged nodes
- Multiple driver files to apply shear from point load and edge shear

## Installation
```bash
git clone https://github.com/feignOP/FEM-Linearelasticity-Solver
cd FEM-Linearelasticity-Solver
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy meshio matplotlib jupyter
```

## Running the Example

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `elasticity.ipynb`  
3. Run all cells:
   - Loads mesh and boundary data from `DataFiles/*.npz`
   - Assembles local and global matrices
   - Applies Dirichlet constraints
   - Solves the sparse linear system
   - Writes results to `VTKoutputs/`

## Visualizing Results in ParaView

1. Open ParaView  
2. File → Open  
3. Select `.vtk` from `VTKoutputs/`  
4. Apply and choose Coloring to visualize concentration  

## Data File Structure (.npz)

- Coord – nodal coordinates  
- Connectivity – Q4 element connectivity  
- Constraints – DOF boundary condition mapping  
- Lx, Ly – domain dimensions  


## License

MIT License

Copyright (c) 2025 Anand Mathew
