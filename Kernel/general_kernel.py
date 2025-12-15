"""
Finite Element Diffusion Solver — Global Assembly Routines
Written by: Anand Mathew

This module contains the core functions for assembling global stiffness matrices,
global load vectors, applying constraints, and reconstructing the full solution
for steady-state diffusion problems.
"""

# Opening Rituals
import numpy as np
from typing import Tuple, Dict, Any
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import meshio

import add_paths

from FemFrameWork.create_id_matrix import create_id_matrix
from FemFrameWork.gauss_quadrature import GaussPoints
from FemFrameWork.shape_functions import ShapeFunctions
from FemFrameWork.vector import Vector

from Physics_models.LE_kernel import CalculateLocalMatrices, TMatrix, Get_Elastic_Moduli, Get_Rhob


# Assemble
def Assemble(dofs_per_node: int,
             EleNodes: np.ndarray,
             GlobalID: np.ndarray,
             klocal: np.ndarray,
             rlocal: np.ndarray,
             K_FF: np.ndarray,
             K_FP: np.ndarray,
             K_PP: np.ndarray,
             R_F: np.ndarray,
             R_P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble an element's local stiffness matrix and load vector into the
    partitioned global system (free–free, free–prescribed, and prescribed–prescribed
    blocks).

    Parameters
    ----------
    dofs_per_node : int
        Number of degrees of freedom per node.
    EleNodes : np.ndarray
        Array of node indices belonging to the current element.
    GlobalID : np.ndarray
        Mapping array of shape (n_nodes, dofs_per_node). Each entry defines
        the global equation number:
        - Positive values → free DOFs (active unknowns)
        - Negative values → prescribed DOFs (Dirichlet conditions)
    Klocal : np.ndarray
        Element stiffness matrix of shape (n_ele_dofs, n_ele_dofs).
    rlocal : np.ndarray
        Element load vector of shape (n_ele_dofs, 1).
    K_FF : np.ndarray
        Global stiffness matrix for free–free DOF interactions.
    K_FP : np.ndarray
        Global coupling matrix between free and prescribed DOFs.
    K_PP : np.ndarray
        Global stiffness matrix for prescribed–prescribed DOF interactions.
    R_F : np.ndarray
        Global right-hand-side vector for free DOFs.
    R_P : np.ndarray
        Global right-hand-side vector for prescribed DOFs.

    Returns
    -------
    K_FF : np.ndarray
        Updated global free–free stiffness matrix.
    K_FP : np.ndarray
        Updated global free–prescribed coupling matrix.
    K_PP : np.ndarray
        Updated global prescribed–prescribed stiffness matrix.
    R_F : np.ndarray
        Updated global right-hand-side vector for free DOFs.
    R_P : np.ndarray
        Updated global right-hand-side vector for prescribed DOFs.

    Notes
    -----
    - Negative entries in `GlobalID` correspond to prescribed DOFs.
    - In standard diffusion problems, only K_FF, K_FP, and R_F are required;
      K_PP and R_P are typically omitted for efficiency.
    - The routine supports extension to coupled or interface formulations
      where K_PP and R_P are needed explicitly.
    """
    NodesPerEle = len(EleNodes)
    v_vector = np.zeros((NodesPerEle * dofs_per_node,), dtype=int)

    
    for i in range(dofs_per_node):
        v_vector[i::dofs_per_node] = GlobalID[EleNodes[:NodesPerEle], i]

    # Scatter local matrices into the global ones
    for row in range(len(v_vector)):
        row_ID = v_vector[row]
        for col in range(len(v_vector)):
            col_ID = v_vector[col]

            if row_ID > 0 and col_ID > 0:
                K_FF[row_ID - 1, col_ID - 1] += klocal[row, col]
            elif row_ID > 0 and col_ID < 0:
                K_FP[row_ID - 1, (-col_ID) - 1] += klocal[row, col]
            elif row_ID < 0 and col_ID < 0:
                K_PP[(-row_ID) - 1, (-col_ID) - 1] += klocal[row, col]

        if row_ID > 0:
            R_F[row_ID - 1, 0] += rlocal[row, 0]
        else:
            R_P[(-row_ID) - 1, 0] += rlocal[row, 0]

    return K_FF, K_FP, K_PP, R_F, R_P


# Global Matrix
def CalculateGlobalMatrices(Connectivity: np.ndarray,
                            Coord: np.ndarray,
                            medium_set: Dict[str, Any],
                            dim: int,
                            dofs_per_node: int,
                            EleType: str,
                            GlobalID: np.ndarray,
                            load_type: Dict[str, Any],
                            NCons: int,
                            Nele: int,
                            NEqns: int,
                            NGPTS: int) -> Tuple[lil_matrix, lil_matrix, np.ndarray]:
    """
    Assemble the global stiffness (diffusion) matrices and global load vector
    by looping over all elements in the finite element mesh.

    Parameters
    ----------
    Conectivity : np.ndarray
        Element connectivity matrix of shape (Nele, Nodes_per_Ele) specifying
        the node indices for each element.
    Coord : np.ndarray
        Global nodal coordinates array of shape (Nnodes, dim).
    diffusivity_function : Dict[str, Any]
        Dictionary defining the diffusivity properties; passed to `Get_DMat`.
    dim : int
        Spatial dimension of the problem (1, 2, or 3).
    dofs_per_node : int
        Number of degrees of freedom per node.
    EleType : str
        Element type identifier (e.g., "L2", "Q4", etc.).
    GlobalID : np.ndarray
        Global degree-of-freedom mapping for each node and DOF.
        Positive entries correspond to free DOFs, negative to prescribed ones.
    load_type : Dict[str, Any]
        Dictionary describing the volumetric source term; passed to `Get_VolumetricSource`.
    NCons : int
        Number of prescribed (constrained) degrees of freedom.
    Nele : int
        Total number of elements in the mesh.
    NEqns : int
        Number of global free equations (size of the free DOF system).
    NGPTS : int
        Number of Gauss integration points per element.

    Returns
    -------
    K_FF : lil_matrix
        Global free–free stiffness matrix assembled from all elements.
    K_FP : lil_matrix
        Global free–prescribed stiffness coupling matrix.
    R_F : np.ndarray
        Global right-hand side vector for free DOFs.

    Notes
    -----
    - Uses `CalculateLocalMatrices` to compute element matrices and
      `Assemble` to add them into the global matrices.
    - Supports both isotropic and anisotropic diffusivity definitions.
    """
    
    # Initialization
    K_FF = lil_matrix((NEqns, NEqns), dtype=float)
    K_FP = lil_matrix((NEqns, NCons), dtype=float)
    K_PP = lil_matrix((NCons, NCons), dtype=float)
    R_F  = np.zeros((NEqns, 1), dtype=float)
    R_P  = np.zeros((NCons, 1), dtype=float)

    NodesPerEle = Connectivity.shape[1]

    # Get Gauss points and weights
    from FemFrameWork.gauss_quadrature import GaussPoints
    r, w = GaussPoints(dim, EleType, NGPTS)

    # Loop over all the elements
    xCap = np.zeros((NodesPerEle, Coord.shape[1]), dtype=float)

    for ele in range(Nele):
        EleNodes_1based = Connectivity[ele, :]
        EleNodes = EleNodes_1based.astype(int) - 1 
        xCap[:NodesPerEle, :] = Coord[EleNodes, :]

        Klocal, rlocal = CalculateLocalMatrices(
            medium_set,
            dofs_per_node,
            EleNodes,
            EleType,
            load_type,
            r,
            w,
            xCap
        )

        K_FF, K_FP, K_PP, R_F, R_P = Assemble(
            dofs_per_node,
            EleNodes,
            GlobalID,
            Klocal,
            rlocal,
            K_FF,
            K_FP,
            K_PP,
            R_F,
            R_P
        )

    # convert to CSR for solving
    return (
        K_FF.tocsr(),
        K_FP.tocsr(),
        K_PP.tocsr(),
        R_F,
        R_P,
    )


# Constraint matrix
def Create_ConstraintsVector(Constraints: np.ndarray, Global_ID: np.ndarray) -> np.ndarray:
    """
    Construct the global prescribed displacement (constraint) vector U_P
    based on the given constraint definitions and global ID mapping.

    Parameters
    ----------
    Constraints : np.ndarray
        Array of shape (NCons, 3) where each row defines a constraint as:
        [node_index, dof_index, prescribed_value].
    Global_ID : np.ndarray
        Global degree-of-freedom mapping array where negative entries correspond
        to prescribed DOFs. Used to locate where each constraint belongs in U_P.

    Returns
    -------
    U_P : np.ndarray
        Global vector of prescribed displacement values of shape (NCons, 1).

    Raises
    ------
    RuntimeError
        If a constraint corresponds to a positive (free) global ID,
        indicating an error in the Global_ID setup.
    """

    # Initializations
    NCons = Constraints.shape[0]
    U_P = np.zeros((NCons, 1))

    for i in range(NCons):
        # convert 1-based to 0-based
        node = int(Constraints[i, 0]) - 1
        dof  = int(Constraints[i, 1]) - 1

        # safety: check dof is valid
        if dof < 0 or dof >= Global_ID.shape[1]:
            raise IndexError(
                f"Constraint {i}: DOF {dof+1} was given, but problem has only {Global_ID.shape[1]} DOF(s)/node."
            )

        constraint_id = Global_ID[node, dof]

        if constraint_id > 0:
            # means you tried to prescribe a DOF that is actually free
            raise RuntimeError("Possible error in Global_ID: constraint points to a free DOF.")

        # constraint_id is negative, so map it to position in U_P
        U_P[abs(constraint_id) - 1, 0] = Constraints[i, 2]

    return U_P


# Post processing
def PostProcessing(Global_ID: np.ndarray, U_F: np.ndarray, U_P: np.ndarray) -> np.ndarray:
    """
    Reconstruct the complete global displacement (solution) matrix U
    from the free (U_F) and prescribed (U_P) displacement vectors.

    Parameters
    ----------
    Global_ID : np.ndarray
        Mapping array of shape (NumNodes, dofs_per_node) that associates each
        node and DOF with its corresponding global equation number:
        - Positive values → index in U_F (free DOFs).
        - Negative values → index in U_P (prescribed DOFs).
    U_F : np.ndarray
        Vector of free DOF displacements, typically the solution of the reduced system.
    U_P : np.ndarray
        Vector of prescribed (constrained) displacements.

    Returns
    -------
    U : np.ndarray
        Full global displacement matrix of shape (NumNodes, dofs_per_node),
        containing both free and prescribed displacement values.

    Notes
    -----
    - Assumes Global_ID uses 1-based numbering for DOF indices.
    - Each entry in U is assigned from U_F or U_P based on the sign of Global_ID.
    """
    
    # Initialization
    NumNodes, dofs_per_node = Global_ID.shape
    U = np.zeros((NumNodes, dofs_per_node))

    # Construction of U matrix
    for node in range(NumNodes):
        for dof in range(dofs_per_node):
            ID = Global_ID[node, dof]
            if ID > 0:
                U[node, dof] = U_F[ID - 1]

            else:
                U[node, dof] = U_P[abs(ID) - 1]
    return U




