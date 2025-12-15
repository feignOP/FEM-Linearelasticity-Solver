import numpy as np
from typing import Dict, Any
from FemFrameWork.gauss_quadrature import GaussPoints
from FemFrameWork.shape_functions import ShapeFunctions
from Physics_models.LE_kernel import Get_Elastic_Moduli


def Stress_Recovery_2D(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    EleType: str,
    medium_set: Dict[str, Any],
    NGPTS: int,
    recovery_type: str,
    U: np.ndarray,
) -> np.ndarray:
    """
    2D stress recovery (plane strain) for linear elasticity.

    Parameters
    ----------
    Connectivity : (Nele, nen)
        Element connectivity (0-based or 1-based; we will auto-fix).
    Coord : (NumNodes, 2)
        Nodal coordinates (x,y).
    EleType : str
        Element type ("Q4", "Q8", "T3", etc.).
    medium_set : dict
        Passed to Get_Elastic_Moduli (returns lambda, mu).
    NGPTS : int
        Number of Gauss points used for GPT recovery.
    recovery_type : {"GPT","EleNodes","AvgNodes"}
        Type of recovery.
    U : (NumNodes, 2)
        Nodal displacements [ux, uy].

    Returns
    -------
    stress_out : ndarray
        For GPT   : shape (Nele*NGPTS, 5)  → [x, y, σxx, σyy, σxy]
        For others: shape (Npts, 5)        → [x, y, σxx, σyy, σxy]
    """
    dim = Coord.shape[1]
    if dim != 2:
        raise ValueError("Stress_Recovery_2D expects Coord with 2 columns (2D).")

    if recovery_type.lower() == "gpt":
        return Stress_Recovery_2D_GPT(Connectivity, Coord, EleType, medium_set, NGPTS, U)
    elif recovery_type.lower() == "elenodes":
        return Stress_Recovery_2D_EleNodes(Connectivity, Coord, EleType, medium_set, U)
    elif recovery_type.lower() == "avgnodes":
        return Stress_Recovery_2D_AvgNodes(Connectivity, Coord, EleType, medium_set, U)
    else:
        raise ValueError(f"Unknown recovery_type: {recovery_type}")


def _build_B_matrix(dNdx: np.ndarray) -> np.ndarray:
    """
    Build 2D (plane strain) B-matrix from dN/dx, dN/dy.

    Parameters
    ----------
    dNdx : (nen, 2)
        Derivatives of shape functions w.r.t x and y.

    Returns
    -------
    B : (3, 2*nen)
        Strain-displacement matrix.
    """
    nen = dNdx.shape[0]
    B = np.zeros((3, 2 * nen), dtype=float)

    for i in range(nen):
        dN_dx = dNdx[i, 0]
        dN_dy = dNdx[i, 1]

        B[0, 2 * i]     = dN_dx        # εxx
        B[1, 2 * i + 1] = dN_dy        # εyy
        B[2, 2 * i]     = dN_dy        # γxy
        B[2, 2 * i + 1] = dN_dx

    return B


def _C_plane_strain(lam: float, mu: float) -> np.ndarray:
    """
    Plane strain constitutive matrix from lambda, mu.
    """
    C = np.array([
        [lam + 2.0 * mu, lam,           0.0],
        [lam,            lam + 2.0 * mu,0.0],
        [0.0,            0.0,           mu ]
    ], dtype=float)
    return C


def Stress_Recovery_2D_GPT(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    EleType: str,
    medium_set: Dict[str, Any],
    NGPTS: int,
    U: np.ndarray,
) -> np.ndarray:
    """
    2D stresses at Gauss points (plane strain).
    Returns (Nele*NGPTS, 5): [x, y, σxx, σyy, σxy].
    """

    # --- handle possible 1-based connectivity ---
    conn = Connectivity.astype(int).copy()
    if conn.min() >= 1:
        conn -= 1

    Nele, nen = conn.shape
    dim = Coord.shape[1]
    r, w = GaussPoints(dim, EleType, NGPTS)
    n_gpt = len(w)

    stress_GPT = np.zeros((Nele * n_gpt, 5), dtype=float)

    for e in range(Nele):
        EleNodes = conn[e, :]
        xCap = Coord[EleNodes, :]      # (nen, 2)
        uCap = U[EleNodes, :]          # (nen, 2)

        for g in range(n_gpt):
            # local coordinates
            if r.ndim == 1:  # (for safety; Q4 should be 2D)
                zeta = np.array([[r[g]]], dtype=float)
            else:
                zeta = r[g, :].reshape(1, -1)    # (1,2)

            N, DN = ShapeFunctions(EleType, zeta)   # N: (1,nen), DN: (nen,2) in ξ,η
            N = np.asarray(N).reshape(-1)           # (nen,)
            J = xCap.T @ DN                         # (2,2)
            invJ = np.linalg.inv(J)
            dNdx = DN @ invJ                        # (nen,2) in x,y

            B = _build_B_matrix(dNdx)               # (3, 2*nen)
            u_e = uCap.reshape(-1)                  # (2*nen,)
            eps = B @ u_e                           # (3,)

            # material
            x_phys = xCap.T @ N                     # (2,)
            lam, mu = Get_Elastic_Moduli(medium_set, x_phys)
            C = _C_plane_strain(lam, mu)
            sig = C @ eps                           # (3,) σxx, σyy, σxy

            idx = e * n_gpt + g
            stress_GPT[idx, 0] = x_phys[0]          # x
            stress_GPT[idx, 1] = x_phys[1]          # y
            stress_GPT[idx, 2:] = sig               # σxx, σyy, σxy

    return stress_GPT


def Stress_Recovery_2D_EleNodes(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    EleType: str,
    medium_set: Dict[str, Any],
    U: np.ndarray,
) -> np.ndarray:
    """
    2D stresses at element nodes (reference-node positions).
    Returns (Nele*nen, 5): [x, y, σxx, σyy, σxy].
    """

    conn = Connectivity.astype(int).copy()
    if conn.min() >= 1:
        conn -= 1

    Nele, nen = conn.shape
    dim = Coord.shape[1]

    # nodal positions in reference space (for Q4: (-1,-1),(1,-1),(1,1),(-1,1))
    # We reuse ShapeFunctions but define a local array of reference points:
    if EleType.lower() in ("q4", "quad4"):
        ref_pts = np.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0,  1.0],
        ])
    else:
        raise NotImplementedError("Stress_Recovery_2D_EleNodes is implemented here only for Q4.")

    stress_EleNodes = np.zeros((Nele * nen, 5), dtype=float)

    for e in range(Nele):
        EleNodes = conn[e, :]
        xCap = Coord[EleNodes, :]   # (nen,2)
        uCap = U[EleNodes, :]       # (nen,2)

        for a in range(nen):
            zeta = ref_pts[a, :].reshape(1, -1)   # (1,2)
            N, DN = ShapeFunctions(EleType, zeta)
            N = np.asarray(N).reshape(-1)        # (nen,)

            J = xCap.T @ DN                      # (2,2)
            invJ = np.linalg.inv(J)
            dNdx = DN @ invJ                     # (nen,2)

            B = _build_B_matrix(dNdx)            # (3,2*nen)
            u_e = uCap.reshape(-1)
            eps = B @ u_e                        # (3,)

            x_phys = xCap.T @ N                  # (2,)
            lam, mu = Get_Elastic_Moduli(medium_set, x_phys)
            C = _C_plane_strain(lam, mu)
            sig = C @ eps

            idx = e * nen + a
            stress_EleNodes[idx, 0] = x_phys[0]
            stress_EleNodes[idx, 1] = x_phys[1]
            stress_EleNodes[idx, 2:] = sig

    return stress_EleNodes


def Stress_Recovery_2D_AvgNodes(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    EleType: str,
    medium_set: Dict[str, Any],
    U: np.ndarray,
) -> np.ndarray:
    """
    2D nodal stress by averaging contributions from connected elements.
    Returns (NumNodes, 5): [x, y, σxx, σyy, σxy].
    """

    conn = Connectivity.astype(int).copy()
    if conn.min() >= 1:
        conn -= 1

    Nele, nen = conn.shape
    NumNodes = Coord.shape[0]

    # 1) get stresses at element nodes
    stress_EleNodes = Stress_Recovery_2D_EleNodes(Connectivity, Coord, EleType, medium_set, U)

    # 2) accumulate and average to global nodes
    stress_Nodes = np.zeros((NumNodes, 5), dtype=float)
    counts = np.zeros(NumNodes, dtype=int)

    # fill coordinates
    stress_Nodes[:, 0:2] = Coord

    for e in range(Nele):
        for a_local in range(nen):
            node = conn[e, a_local]          # 0-based
            idx_ele = e * nen + a_local
            stress_Nodes[node, 2:] += stress_EleNodes[idx_ele, 2:]
            counts[node] += 1

    # avoid division by zero
    mask = counts > 0
    stress_Nodes[mask, 2:] /= counts[mask, None]

    return stress_Nodes