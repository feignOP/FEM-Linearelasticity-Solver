import numpy as np
from typing import Dict, Any
from FemFrameWork.gauss_quadrature import GaussPoints
from FemFrameWork.shape_functions import ShapeFunctions
from Physics_models.LE_kernel import Get_Elastic_Moduli


def Stress_Recovery(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    EleType: str,
    medium_set: Dict[str, Any],
    NGPTS: int,
    recovery_type: str,
    U: np.ndarray
) -> np.ndarray:

    if recovery_type.lower() == "gpt":
        return Stress_Recovery_GPT(Connectivity, Coord, EleType, medium_set, NGPTS, U)
    elif recovery_type.lower() == "elenodes":
        return Stress_Recovery_EleNodes(Connectivity, Coord, EleType, medium_set, U)
    elif recovery_type.lower() == "avgnodes":
        return Stress_Recovery_AvgNodes(Connectivity, Coord, EleType, medium_set, U)
    else:
        raise ValueError("Unknown recovery_type: " + recovery_type)


def Stress_Recovery_GPT(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    EleType: str,
    medium_set: Dict[str, Any],
    NGPTS: int,
    U: np.ndarray
) -> np.ndarray:

    dim = Coord.shape[1]
    Nele = Connectivity.shape[0]
    r, w = GaussPoints(dim, EleType, NGPTS)
    stress_GPT = np.zeros((Nele * len(w), 2), dtype=float)

    for ele in range(Nele):
        EleNodes = Connectivity[ele]
        xCap = Coord[EleNodes]
        if xCap.ndim == 1:
            xCap = xCap.reshape(-1, 1)
        uCap = U[EleNodes]
        if uCap.ndim == 1:
            uCap = uCap.reshape(-1, 1)

        for gpt in range(len(w)):
            zeta = np.array([r[gpt]]) if r.ndim == 1 else r[gpt, :].reshape(1, -1)
            N, DN = ShapeFunctions(EleType, zeta)
            N = np.array(N).flatten()
            J = xCap.T @ DN
            if J.size == 1:
                B = DN / J
            else:
                B = DN @ np.linalg.inv(J)
            x = float(xCap.T @ N)
            gradu = (B.T @ uCap).squeeze()
            strain = float(gradu)
            lam, mu = Get_Elastic_Moduli(medium_set, np.array([x]))
            E = mu * (3 * lam + 2 * mu) / (lam + mu)
            stress_GPT[NGPTS * ele + gpt, 0] = x
            stress_GPT[NGPTS * ele + gpt, 1] = E * strain

    return stress_GPT


def Stress_Recovery_EleNodes(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    EleType: str,
    medium_set: Dict[str, Any],
    U: np.ndarray
) -> np.ndarray:

    Nele = Connectivity.shape[0]
    NodesPerEle = Connectivity.shape[1]
    stress_EleNodes = np.zeros((Nele * NodesPerEle, 2), dtype=float)
    r = np.array([-1.0, 1.0], dtype=float)

    for ele in range(Nele):
        EleNodes = Connectivity[ele]
        xCap = Coord[EleNodes]
        if xCap.ndim == 1:
            xCap = xCap.reshape(-1, 1)
        uCap = U[EleNodes]
        if uCap.ndim == 1:
            uCap = uCap.reshape(-1, 1)

        for gpt in range(NodesPerEle):
            zeta = np.array([r[gpt]]) if r.ndim == 1 else r[gpt, :].reshape(1, -1)
            N, DN = ShapeFunctions(EleType, zeta)
            N = np.array(N).flatten()
            J = xCap.T @ DN
            if J.size == 1:
                B = DN / J
            else:
                B = DN @ np.linalg.inv(J)
            x = float(xCap.T @ N)
            gradu = (B.T @ uCap).squeeze()
            strain = float(gradu)
            lam, mu = Get_Elastic_Moduli(medium_set, np.array([x]))
            E = mu * (3 * lam + 2 * mu) / (lam + mu)
            stress_EleNodes[NodesPerEle * ele + gpt, 0] = x
            stress_EleNodes[NodesPerEle * ele + gpt, 1] = E * strain

    return stress_EleNodes


def Stress_Recovery_AvgNodes(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    EleType: str,
    medium_set: Dict[str, Any],
    U: np.ndarray
) -> np.ndarray:

    Nele = Connectivity.shape[0]
    NumNodes = Coord.shape[0]
    stress_EleNodes = Stress_Recovery_EleNodes(Connectivity, Coord, EleType, medium_set, U)
    stress_Nodes = np.zeros((NumNodes, 2), dtype=float)
    stress_Nodes[:, 0] = Coord[:, 0]

    for ele in range(Nele):
        stress_Nodes[ele, 1]     += stress_EleNodes[2 * ele, 1]
        stress_Nodes[ele + 1, 1] += stress_EleNodes[2 * ele + 1, 1]

    if NumNodes > 2:
        stress_Nodes[1:NumNodes - 1, 1] *= 0.5

    return stress_Nodes