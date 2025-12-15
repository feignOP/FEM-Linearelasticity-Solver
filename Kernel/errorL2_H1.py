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


def Calculate_Error(Connectivity: np.ndarray,
                    Coord: np.ndarray,
                    EleType: str,
                    NGPTS: int,
                    U: np.ndarray,
                    exact_solution,
                    exact_gradient) -> tuple[float, float]:
    """
    Compute L2 and H1-seminorm errors of the FEM solution.

    Parameters
    ----------
    Connectivity : (Nele, nen)
        Element connectivity array (0-based node indices).
    Coord : (Nnodes, dim)
        Global coordinates of all nodes.
    EleType : str
        Element type (e.g., "Q4", "T3", etc.).
    NGPTS : int
        Number of Gauss points per direction.
    U : (Nnodes, 1)
        Nodal solution vector (including prescribed and free DOFs).
    exact_solution : callable
        Function handle f(x, y) returning the analytical (exact) solution.
    exact_gradient : callable
        Function handle grad_f(x, y) returning ∇u_exact = [du/dx, du/dy].

    Returns
    -------
    error_in_L2 : float
        L2 norm of the error.
    error_in_H1_seminorm : float
        H1 seminorm of the error.
    """

    # Initializing
    error_in_L2 = 0.0
    error_in_H1 = 0.0

    if Connectivity.min() == 1:
        Connectivity = Connectivity - 1

    Nele = Connectivity.shape[0]
    dim = Coord.shape[1]

    U = np.asarray(U).reshape(-1)

    # Gauss points
    r, w = GaussPoints(dim, EleType, NGPTS)

    for ele in range(Nele):
        EleNodes = Connectivity[ele, :]  

        # nodal values for this element
        uCap = U[EleNodes].reshape(-1, 1)   
        xCap = Coord[EleNodes, :]              
        for gpt in range(len(w)):
            zeta = np.array([r[gpt]]) if r.ndim == 1 else r[gpt, :].reshape(1, -1)

            N, DN = ShapeFunctions(EleType, zeta)
            N = np.array(N).flatten()        

            x = (xCap.T @ N).flatten()       
            J = xCap.T @ DN
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            B = DN @ invJ                      

            # interpolate u
            u = float(N @ uCap.squeeze())
            gradu = (B.T @ uCap).squeeze()   

            # exact
            uExact = float(exact_solution(*x))
            duExact = np.asarray(exact_gradient(*x))

            # L2 
            error_in_L2 += w[gpt] * (u - uExact) ** 2 * detJ

            # H1 
            delGrad = gradu - duExact[:dim]
            error_in_H1 += w[gpt] * (delGrad @ delGrad) * detJ

    return np.sqrt(error_in_L2), np.sqrt(error_in_H1)