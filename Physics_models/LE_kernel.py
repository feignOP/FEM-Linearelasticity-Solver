"""
Finite Element Diffusion Solver — Diffusion Kernel Routines
Written by: Anand Mathew
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



import numpy as np

def TMatrix(m, n):
    """
    Construct the transpose-operator matrix T used in linear elasticity.

    This matrix satisfies:
        vec(A.T) = T @ vec(A)
    for any (m x n) matrix A, where vec() is column-stacking vectorization.

    Parameters
    ----------
    m : int
        Number of rows of the original matrix A.
    n : int
        Number of columns of the original matrix A.

    Returns
    -------
    T : ndarray of shape (m*n, m*n)
        The permutation matrix that maps vec(A) → vec(A.T).
    """
    Im = np.eye(m)
    In = np.eye(n)
    T = np.zeros((m * n, m * n))
    for i in range(n):
        T[i * m:(i + 1) * m, :] = np.kron(Im, In[i, :])
    return T



# Elasticity Tensor
def Get_Elastic_Moduli(medium_set: Dict[str, Any], x: np.ndarray = None) -> Tuple[float, float]:
    """
    Return the elastic moduli (lambda, mu) for isotropic elasticity.

    Parameters
    ----------
    medium_set : dict
        Must contain:
            - "type": must be "Lame_params"
            - "lambda": first Lame parameter (float)
            - "mu": shear modulus (float)
    x : np.ndarray, optional
        Spatial point (not used; present for interface compatibility).

    Returns
    -------
    (lambda, mu) : tuple of floats

    Raises
    ------
    ValueError
        If the medium_set type is not supported.
    """
    mtype = str(medium_set.get("type", "")).lower()

    if mtype == "lame_params":
        lam = medium_set.get("lambda")
        mu  = medium_set.get("mu")

        if lam is None or mu is None:
            raise ValueError("medium_set must contain 'lambda' and 'mu'.")

        return float(lam), float(mu)

    if mtype == "elastic_moduli":
        E = medium_set.get("E")
        lam = 0.0
        mu  = 0.5 * float(E)
        return lam, mu

    raise ValueError(f"Given medium_set '{medium_set.get('type')}' is not available.")

# Body force

def Get_Rhob(load_type: Dict[str, Any], x: np.ndarray, dim: int) -> np.ndarray:
    """
    Return the body force vector f for gravity loading.

    Parameters
    ----------
    load_type : dict
        Must contain:
            - "case": either "2D_gravity" or "3D_gravity"
            - "rhob": scalar (e.g. rho * g)
    x : np.ndarray
        Spatial point (not used here, kept for interface compatibility).
    dim : int
        Spatial dimension (ignored; determined from 'case').

    Returns
    -------
    np.ndarray
        Body force vector f of length 2 or 3.

    Raises
    ------
    ValueError
        If the case is not recognized.
    """
    case = str(load_type.get("case", "")).lower()
    rhob = load_type.get("rhob")

    if rhob is None:
        raise ValueError("load_type must contain a 'rhob' entry")

    if case == "1d_gravity":
        f = np.zeros(1)
        # In 1D rod, convention: negative = downward
        f[0] = -rhob
        return f


    if case == "2d_gravity":
        f = np.zeros(2)
        f[1] = -rhob
        return f

    if case == "3d_gravity":
        f = np.zeros(3)
        f[2] = -rhob
        return f

    raise ValueError(f"Given body-force case '{load_type.get('case')}' is not supported.")


# Calculate Local matrix
def CalculateLocalMatrices(medium_set: Dict[str, Any],
                           dofs_per_node: int,
                           EleNodes: np.ndarray,
                           EleType: str,
                           load_type: Dict[str, Any],
                           r: np.ndarray,
                           w: np.ndarray,
                           xCap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble the element-level stiffness (diffusion) matrix and load vector
    for a single finite element using Gaussian quadrature.

    Parameters
    ----------
    diffusivity_function : Dict[str, Any]
        Description of the diffusivity; passed to `Get_DMat` to obtain D(x).
    dofs_per_node : int
        Number of degrees of freedom per node (1 for scalar problems).
    EleNodes : np.ndarray
        Node indices of the current element (not directly used here but kept
        for interface completeness).
    EleType : str
        Element type identifier, passed to `ShapeFunctions` (e.g. "L2", "Q4").
    load_type : Dict[str, Any]
        Description of the volumetric source; passed to `Get_VolumetricSource`.
    r : np.ndarray
        Gauss point locations in the reference element (1D, 2D, or 3D array).
    w : np.ndarray
        Corresponding Gauss weights.
    xCap : np.ndarray
        Nodal coordinates of the element in physical space, shape (n_nodes, dim).

    Returns
    -------
    klocal : np.ndarray
        Local element stiffness/diffusion matrix of shape
        (n_nodes * dofs_per_node, n_nodes * dofs_per_node).
    rlocal : np.ndarray
        Local element load vector of shape (n_nodes * dofs_per_node, 1).

    Notes
    -----
    - Supports both scalar D (isotropic) and full matrix D (anisotropic).
    - For multiple dofs per node, expands the scalar/matrix part using kron(I, Ke).
    """
    
    # Initialization
    NodesPerEle = len(EleNodes)
    dim = xCap.shape[1]
    klocal = np.zeros((NodesPerEle * dofs_per_node,
                       NodesPerEle * dofs_per_node))
    rlocal = np.zeros((NodesPerEle * dofs_per_node, 1))
    
    xCap = np.array(xCap, dtype=float)
    if xCap.ndim == 1:
        xCap = xCap.reshape(-1, 1)

    # Loop over Gauss points
    for gpt in range(len(w)):

        zeta = np.array([r[gpt]]) if r.ndim == 1 else r[gpt, :].reshape(1, -1)

        # N and DN from shape functions
        N, DN = ShapeFunctions(EleType, zeta)
        N = np.array(N).flatten()

        J = xCap.T @ DN
        detJ = np.linalg.det(J)
        B = DN @ np.linalg.inv(J)
        x = xCap.T @ N      

        lam, mu = Get_Elastic_Moduli(medium_set, x)   

        # klocal
        klocal += w[gpt] * lam * (Vector(B.T) @ Vector(B.T).T) * detJ

        klocal += w[gpt] * mu * (np.kron(B, np.eye(dim)) @ np.kron(B.T, np.eye(dim))) * detJ

        klocal = klocal + w[gpt] * mu * (np.kron(B, np.eye(dim)) @ TMatrix(dim, dim) @ np.kron(B.T, np.eye(dim))) * detJ

        
        # load term contribution
        if dim == 1:
            f_body = Get_Rhob(load_type, x, dim)   # shape (1,)
            rlocal[:, 0] += w[gpt] * N * f_body[0] * detJ
        else:
            f_body = Get_Rhob(load_type, x, dim)
            f_body = np.asarray(f_body).reshape(-1)

            N_vec = np.array(N).flatten().reshape(-1, 1)
            BF = np.kron(N_vec, np.eye(dim))

            rlocal[:, 0] += w[gpt] * (BF @ f_body) * detJ
        
        #else:
            #rlocal += w[gpt] * np.kron(N.T, np.eye(dim)) @ Get_Rhob(load_type, x, dim).reshape(-1, 1) * detJ

        

    return klocal, rlocal




