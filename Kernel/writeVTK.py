"""
Module to write VTK files from finite element data.
"""

# Opening rituals
from typing import Dict, Any, Tuple, Optional
import numpy as np
import meshio

def write_vtk(vtk_filename: str,
              Coord: np.ndarray,
              Connectivity: np.ndarray,
              EleType: str,
              U: np.ndarray) -> None:

    ELETYPE_TO_MESHIO = {
        "L2": "line", "L3": "line3",
        "T3": "triangle", "TRI3": "triangle", "T6": "triangle6",
        "Q4": "quad", "QUAD4": "quad", "Q8": "quad8", "Q9": "quad9",
        "H8": "hexahedron", "HEX8": "hexahedron", "B8": "hexahedron",
        "TET4": "tetra", "TETRA4": "tetra",
    }

    # Promote 2D coords to 3D (VTK needs 3D points)
    pts = Coord
    if pts.shape[1] == 2:
        pts = np.column_stack((pts, np.zeros(pts.shape[0])))

    # --- Auto-fix indexing: convert 1-based -> 0-based if needed ---
    conn = np.asarray(Connectivity, dtype=int).copy()
    num_nodes = pts.shape[0]
    if conn.min() >= 1:
        conn -= 1  # normalize to 0-based
    # sanity: now all indices must be in [0, num_nodes-1]
    if conn.min() < 0 or conn.max() >= num_nodes:
        raise ValueError(
            f"Connectivity indices out of range after normalization: "
            f"min={conn.min()}, max={conn.max()}, num_nodes={num_nodes}"
        )

    # Resolve cell type
    ele_key = EleType.upper()
    if ele_key in ELETYPE_TO_MESHIO:
        cell_type = ELETYPE_TO_MESHIO[ele_key]
    else:
        nen = conn.shape[1]
        if nen == 4:
            cell_type = "quad"
        elif nen == 3:
            cell_type = "triangle"
        elif nen == 2:
            cell_type = "line"
        else:
            raise ValueError(f"Unsupported element type: {EleType}")

    # Decide whether U is scalar (diffusion) or vector (elasticity)
    U_arr = np.asarray(U)

    if U_arr.ndim == 1:
        # scalar field
        point_data = {"U": U_arr.ravel()}
    elif U_arr.ndim == 2 and U_arr.shape[1] == 1:
        point_data = {"U": U_arr.ravel()}
    else:
        ndof = U_arr.shape[1]
        U_vec = np.zeros((num_nodes, 3), dtype=float)
        if ndof >= 3:
            U_vec[:, :3] = U_arr[:, :3]
        elif ndof == 2:
            U_vec[:, 0] = U_arr[:, 0]   
            U_vec[:, 1] = U_arr[:, 1] 
        else:  
            U_vec[:, 0] = U_arr[:, 0]
        point_data = {"U": U_vec}

    mesh = meshio.Mesh(
        points=pts,
        cells=[(cell_type, conn)],
        point_data=point_data,
    )
    mesh.write(vtk_filename, file_format="vtk")