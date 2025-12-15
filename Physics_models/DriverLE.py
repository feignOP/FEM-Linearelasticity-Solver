# Opening rituals
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Any
from scipy.sparse import lil_matrix, csr_matrix

from scipy.sparse.linalg import spsolve
import meshio

import sys
from pathlib import Path

import inspect

from IPython.display import Image, display


import add_paths

from FemFrameWork.create_id_matrix import create_id_matrix
from FemFrameWork.gauss_quadrature import GaussPoints
from FemFrameWork.shape_functions import ShapeFunctions
from FemFrameWork.vector import Vector

from Physics_models.LE_kernel import CalculateLocalMatrices, TMatrix, Get_Elastic_Moduli, Get_Rhob
from Kernel.general_kernel import CalculateGlobalMatrices, Assemble, Create_ConstraintsVector, PostProcessing
from Kernel.writeVTK import write_vtk
from Kernel.stress_recovery import Stress_Recovery

import importlib


def Driver_LE(Coord: np.ndarray, Connectivity: np.ndarray, constraints: np.ndarray,
                            medium_set: Dict[str, Any], load_type: Dict[str, Any], EleType: str,
                            dofs_per_node: int, dim: int, NGPTS: int, vtk_filename: str = "VTKoutputs/solution.vtk") -> np.ndarray:
    """
    Final FEM pipeline for quasistatic linear elasticity.

    Parameters
    ----------
    Coord
    Connectivity 
    constraints : (NCons, 3)
    diffusivity_function : dict
        Passed straight to Get_DMat inside CalculateLocalMatrices.
    load_type : dict
        Passed straight to Get_VolumetricSource.
    EleType : str
        Element type (e.g., "Q4").
    dofs_per_node : int
        DOFs per node (1 for diffusion problems).
    dim : int
        Spatial dimension (1, 2, or 3).
    NGPTS : int
        Number of Gauss points to use per element.
    vtk_filename : str, optional
        Output filename for VTK export (default: "solution.vtk").

    Returns
    -------
    U : (NumNodes, dofs_per_node)
        Full nodal solution (free + prescribed).
    """
    # Basic derived quantities
    NumNodes = Coord.shape[0]
    Nele = Connectivity.shape[0]
    NCons = constraints.shape[0]

    print("──────── Simulation status report ────────")
    print(f"Nodes={NumNodes}, Elems={Nele}, DOFs/node={dofs_per_node}")

    # Create GlobalID, NEqns
    GlobalID, NEqns = create_id_matrix(constraints, dofs_per_node, NumNodes)
    print(f"[1/6] GlobalID created. NEqns = {NEqns}, NCons = {NCons}")

    # Constraint
    U_P = Create_ConstraintsVector(constraints, GlobalID)
    print("[2/6] U_P created.")

    # Create global matrices 
    K_FF, K_FP, K_PP, R_F, R_P = CalculateGlobalMatrices(
        Connectivity=Connectivity,
        Coord=Coord,
        medium_set=medium_set,
        dim=dim,
        dofs_per_node=dofs_per_node,
        EleType=EleType,
        GlobalID=GlobalID,
        load_type=load_type,
        NCons=NCons,
        Nele=Nele,
        NEqns=NEqns,
        NGPTS=NGPTS,
    )
    print("[3/6] Global matrices assembled.")

    # Solve
    RHS = R_F - K_FP @ U_P
    U_F = spsolve(K_FF, RHS)
    print("[4/6] Linear system solved.")

    # Post processing 
    U = PostProcessing(GlobalID, U_F, U_P)
    print("[5/6] Postprocessing done.")

    # Write VTK output
    write_vtk(vtk_filename, Coord, Connectivity, EleType, U)
    print(f"[6/6] VTK written to {vtk_filename}")

    print("FEM simulation complete!")
    return U


def Driver_LE_pointload(Coord: np.ndarray, Connectivity: np.ndarray, constraints: np.ndarray,
                            medium_set: Dict[str, Any], load_type: Dict[str, Any], EleType: str,
                            dofs_per_node: int, dim: int, NGPTS: int, nodepl: int, dof_pl: int, P: float,
                            vtk_filename: str = "VTKoutputs/solution.vtk") -> np.ndarray:
    """
    Final FEM pipeline for quasistatic linear elasticity with a concentrated point load.

    Parameters
    ----------
    Coord : (NumNodes, dim) ndarray
        Nodal coordinates of the mesh.
    Connectivity : (Nele, nen) ndarray
        Element connectivity using one based node numbering.
    constraints : (NCons, 3) ndarray
        Essential boundary conditions given as
        [node_id, dof_id, prescribed_value].
    medium_set : dict
        Material model parameters passed to the elasticity kernel
        for example Lamé parameters.
    load_type : dict
        Volumetric load specification passed to the element routines.
    EleType : str
        Element type such as Q4.
    dofs_per_node : int
        Number of degrees of freedom per node.
    dim : int
        Spatial dimension of the problem.
    NGPTS : int
        Number of Gauss points per element.
    nodepl : int
        One based node number where the point load is applied.
    dof_pl : int
        Degree of freedom index at the loaded node.
    P : float
        Magnitude of the applied point load.
    vtk_filename : str, optional
        Filename for VTK output.

    Returns
    -------
    U : (NumNodes, dofs_per_node) ndarray
        Complete nodal displacement field including free and prescribed values.
    """
    # Basic derived quantities
    NumNodes = Coord.shape[0]
    Nele = Connectivity.shape[0]
    NCons = constraints.shape[0]

    print("──────── Simulation status report ────────")
    print(f"Nodes={NumNodes}, Elems={Nele}, DOFs/node={dofs_per_node}")

    # Create GlobalID, NEqns
    GlobalID, NEqns = create_id_matrix(constraints, dofs_per_node, NumNodes)
    print(f"[1/6] GlobalID created. NEqns = {NEqns}, NCons = {NCons}")

    # Constraint
    U_P = Create_ConstraintsVector(constraints, GlobalID)
    print("[2/6] U_P created.")

    # Create global matrices 
    K_FF, K_FP, K_PP, R_F, R_P = CalculateGlobalMatrices(
        Connectivity=Connectivity,
        Coord=Coord,
        medium_set=medium_set,
        dim=dim,
        dofs_per_node=dofs_per_node,
        EleType=EleType,
        GlobalID=GlobalID,
        load_type=load_type,
        NCons=NCons,
        Nele=Nele,
        NEqns=NEqns,
        NGPTS=NGPTS,
    )
    print("[3/6] Global matrices assembled.")

    eq = GlobalID[nodepl - 1, dof_pl - 1]
    if eq > 0:                    
        R_F[eq - 1, 0] += P
        print("[3(a)/6] Point load used.")

    # Solve
    RHS = R_F - K_FP @ U_P
    U_F = spsolve(K_FF, RHS)
    print("[4/6] Linear system solved.")

    # Post processing 
    U = PostProcessing(GlobalID, U_F, U_P)
    print("[5/6] Postprocessing done.")

    # Write VTK output
    write_vtk(vtk_filename, Coord, Connectivity, EleType, U)
    print(f"[6/6] VTK written to {vtk_filename}")

    print("FEM simulation complete!")
    return U






    