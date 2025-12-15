"""
Written by: Anand Mathew
Shape functions and their derivatives for various finite elements.
Master function: ShapeFunctions
Sub functions: Line2, Line3, Quad4, Quad8, Tri3, Tri6, Brick8, Tet4, Wedge6
"""

# Opening rituals
import numpy as np
from typing import Tuple


# Shape function master function
def ShapeFunctions(EleType: str, zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """
  Returns the shape functions (N) and their derivatives (DN)
  for the specified element type and local coordinates zeta.

  Parameters
  ----------
  EleType : str
      Element type identifier (e.g., "Q4", "T3", "B8", ...).
  zeta : np.ndarray
      Local coordinates (xi, eta, zeta) depending on the element.

  Returns
  -------
  N : np.ndarray
      Shape function values.
  DN : np.ndarray
      Derivatives of shape functions with respect to local coordinates.

  Raises
  ------
  ValueError
      If input are invalid.
  """
  key = EleType.strip().lower()

  # 1D Elements
  if key == "l2":
      N, DN = Line2(zeta)
  elif key == "l3":
      N, DN = Line3(zeta)

  # 2D Elements
  elif key == "q4":
      N, DN = Quad4(zeta)
  elif key == "q8":
      N, DN = Quad8(zeta)
  elif key == "t3":
      N, DN = Tri3(zeta)
  elif key == "t6":
      N, DN = Tri6(zeta)

  # 3D Elements
  elif key == "b8":
      N, DN = Brick8(zeta)
  elif key == "tet4":
      N, DN = Tet4(zeta)
  elif key == "w6":
      N, DN = Wedge6(zeta)

  else:
      raise ValueError(f"Unknown element type '{EleType}'.")

  return N, DN


def Line2(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates N and DN for Line element with 2 nodes
    """

    if not isinstance(zeta, np.ndarray):
        raise TypeError(f"Expected zeta as np.ndarray, got {type(zeta)}.")
    if zeta.size != 1:
        raise ValueError("Needs to be 1 coordinate!")
    
    zeta = float(zeta)
    
    # N
    N1 = 0.5 * (1 - zeta)
    N2 = 0.5 * (1 + zeta)
    N = np.array([[N1, N2]])

    # DN
    DN = np.array([[-0.5], [0.5]])

    return N, DN


def Line3(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates N and DN for Line element with 3 nodes
    """

    if not isinstance(zeta, np.ndarray):
        raise TypeError(f"Expected zeta as np.ndarray, got {type(zeta)}.")
    if zeta.size != 1:
        raise ValueError("Needs to be 1 coordinate!")
    
    zeta = float(zeta)
    
    # N
    N1 = 0.5 * zeta * (zeta - 1)
    N2 = 0.5 * zeta * (zeta + 1)
    N3 = 1 - zeta**2
    N = np.array([[N1, N2, N3]])

    # DN
    dN1 = zeta - 0.5
    dN2 = zeta + 0.5
    dN3 = -2 * zeta
    DN = np.array([[dN1], [dN2], [dN3]])

    return N, DN


def Quad4(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates N and DN for Quadrilateral element with 4 nodes
    """
    if not isinstance(zeta, np.ndarray):
        raise TypeError(f"Expected zeta as np.ndarray, got {type(zeta)}.")
    if zeta.size != 2:
        raise ValueError("Needs to be two coordinates!")
    
    zeta1 = float(zeta[:, 0])
    zeta2 = float(zeta[:, 1])

    # N
    N1 = 0.25 * (1.0 - zeta1) * (1.0 - zeta2)
    N2 = 0.25 * (1.0 + zeta1) * (1.0 - zeta2)
    N3 = 0.25 * (1.0 + zeta1) * (1.0 + zeta2)
    N4 = 0.25 * (1.0 - zeta1) * (1.0 + zeta2)
    N  = np.array([[N1, N2, N3, N4]])

    # DN
    dN11 = -0.25 * (1.0 - zeta2)
    dN12 = -0.25 * (1.0 - zeta1)
    dN21 =  0.25 * (1.0 - zeta2)
    dN22 = -0.25 * (1.0 + zeta1)
    dN31 =  0.25 * (1.0 + zeta2)
    dN32 =  0.25 * (1.0 + zeta1)
    dN41 = -0.25 * (1.0 + zeta2)
    dN42 =  0.25 * (1.0 - zeta1)

    DN = np.array([[dN11, dN12], [dN21, dN22], [dN31, dN32], [dN41, dN42]])

    return N, DN


def Quad8(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates N and DN for Quadrilateral element with 8 nodes
    """
    if not isinstance(zeta, np.ndarray):
        raise TypeError(f"Expected zeta as np.ndarray, got {type(zeta)}.")
    if zeta.size != 2:
        raise ValueError("Needs to be a two coordinates!")
    
    zeta1 = float(zeta[:, 0])
    zeta2 = float(zeta[:, 1])

    # N
    N1 = 0.25 * (1 - zeta1) * (1 - zeta2) * (-1 - zeta1 - zeta2)
    N2 = 0.25 * (1 + zeta1) * (1 - zeta2) * ( zeta1 - zeta2 - 1)
    N3 = 0.25 * (1 + zeta1) * (1 + zeta2) * ( zeta1 + zeta2 - 1)
    N4 = 0.25 * (1 - zeta1) * (1 + zeta2) * (-zeta1 + zeta2 - 1)

    N5 = 0.5  * (1 - zeta1**2) * (1 - zeta2)
    N6 = 0.5  * (1 + zeta1)    * (1 - zeta2**2)
    N7 = 0.5  * (1 - zeta1**2) * (1 + zeta2)
    N8 = 0.5  * (1 - zeta1)    * (1 - zeta2**2)

    N = np.array([[N1, N2, N3, N4, N5, N6, N7, N8]])

    # DN
    dN11 = 0.25 * (1 - zeta2) * (2*zeta1 + zeta2)
    dN21 = 0.25 * (1 - zeta2) * (2*zeta1 - zeta2)
    dN31 = 0.25 * (1 + zeta2) * (2*zeta1 + zeta2)
    dN41 = 0.25 * (1 + zeta2) * (2*zeta1 - zeta2)
    dN51 = -zeta1 * (1 - zeta2)
    dN61 = 0.5 * (1 - zeta2**2)
    dN71 = -zeta1 * (1 + zeta2)
    dN81 = -0.5 * (1 - zeta2**2)
    dN12 = 0.25 * (1 - zeta1) * (zeta1 + 2*zeta2)
    dN22 = -0.25 * (1 + zeta1) * (zeta1 - 2*zeta2)
    dN32 = 0.25 * (1 + zeta1) * (zeta1 + 2*zeta2)
    dN42 = 0.25 * (1 - zeta1) * (-zeta1 + 2*zeta2)
    dN52 = -0.5 * (1 - zeta1**2)
    dN62 = -(1 + zeta1) * zeta2
    dN72 = 0.5 * (1 - zeta1**2)
    dN82 = -(1 - zeta1) * zeta2

    DN = DN = np.array([[dN11, dN12], 
                        [dN21, dN22], 
                        [dN31, dN32],  
                        [dN41, dN42],  
                        [dN51, dN52],  
                        [dN61, dN62],  
                        [dN71, dN72],  
                        [dN81, dN82], ])

    return N, DN


def Tri3(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates N and DN for triangular element with 3 nodes
    """
    if not isinstance(zeta, np.ndarray):
        raise TypeError(f"Expected zeta as np.ndarray, got {type(zeta)}.")
    if zeta.size != 2:
        raise ValueError("Needs to be two coordinates!")

    zeta1 = float(zeta[:, 0])
    zeta2 = float(zeta[:, 1])

    # N
    N1 = 1.0 - zeta1 - zeta2
    N2 = zeta1
    N3 = zeta2
    N  = np.array([[N1, N2, N3]])

    # DN
    DN = np.array([
        [-1.0,  -1.0],   
        [1.0,  0.0],
        [0.0,  1.0]     
    ])

    return N, DN


def Tri6(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates N and DN for triangular element with 6 nodes
    """
    if not isinstance(zeta, np.ndarray):
        raise TypeError(f"Expected zeta as np.ndarray, got {type(zeta)}.")
    if zeta.size != 2:
        raise ValueError("Needs to be two coordinates!")

    zeta1 = float(zeta[:, 0])
    zeta2 = float(zeta[:, 1])

    # N
    N1 = (1.0 - zeta1 - zeta2) * (1.0 - 2.0 * zeta1 - 2.0 * zeta2)
    N2 = zeta1 * (2.0 * zeta1 - 1.0)
    N3 = zeta2 * (2.0* zeta2 - 1.0)
    N4 = 4.0 * zeta1 * (1.0 - zeta1 - zeta2)
    N5 = 4.0 * zeta1 * zeta2
    N6 = 4.0 * zeta2 * (1.0 - zeta1 - zeta2)
    N  = np.array([[N1, N2, N3, N4, N5, N6]])

    # DN
    dN11 = -3.0 + 4.0*zeta1 + 4.0*zeta2
    dN12 = -3.0 + 4.0*zeta1 + 4.0*zeta2
    dN21 = 4.0*zeta1 - 1.0
    dN22 = 0.0
    dN31 = 0.0
    dN32 = 4.0*zeta2 - 1.0
    dN41 = 4.0*(1.0 - 2.0*zeta1 - zeta2)
    dN42 = -4.0*zeta1
    dN51 = 4.0*zeta2
    dN52 = 4.0*zeta1
    dN61 = -4.0*zeta2
    dN62 = 4.0*(1.0 - zeta1 - 2.0*zeta2)

    # DN matrix (6 x 2)
    DN = np.array([
        [dN11, dN12],
        [dN21, dN22],
        [dN31, dN32],
        [dN41, dN42],
        [dN51, dN52],
        [dN61, dN62]
    ])

    return N, DN


def Brick8(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates N and DN for brick element with 8 nodes
    """
    if not isinstance(zeta, np.ndarray):
        raise TypeError(f"Expected zeta as np.ndarray, got {type(zeta)}.")
    if zeta.size != 3:
        raise ValueError("Needs to be three coordinates!")

    zeta1 = float(zeta[:, 0])
    zeta2 = float(zeta[:, 1])
    zeta3 = float(zeta[:, 2])

    # Shape functions
    N1 = 0.125 * (1 - zeta1) * (1 - zeta2) * (1 - zeta3)
    N2 = 0.125 * (1 + zeta1) * (1 - zeta2) * (1 - zeta3)
    N3 = 0.125 * (1 + zeta1) * (1 + zeta2) * (1 - zeta3)
    N4 = 0.125 * (1 - zeta1) * (1 + zeta2) * (1 - zeta3)
    N5 = 0.125 * (1 - zeta1) * (1 - zeta2) * (1 + zeta3)
    N6 = 0.125 * (1 + zeta1) * (1 - zeta2) * (1 + zeta3)
    N7 = 0.125 * (1 + zeta1) * (1 + zeta2) * (1 + zeta3)
    N8 = 0.125 * (1 - zeta1) * (1 + zeta2) * (1 + zeta3)
    N  = np.array([[N1, N2, N3, N4, N5, N6, N7, N8]])

    # DN
    dN11 = -0.125 * (1 - zeta2) * (1 - zeta3)
    dN21 =  0.125 * (1 - zeta2) * (1 - zeta3)
    dN31 =  0.125 * (1 + zeta2) * (1 - zeta3)
    dN41 = -0.125 * (1 + zeta2) * (1 - zeta3)
    dN51 = -0.125 * (1 - zeta2) * (1 + zeta3)
    dN61 =  0.125 * (1 - zeta2) * (1 + zeta3)
    dN71 =  0.125 * (1 + zeta2) * (1 + zeta3)
    dN81 = -0.125 * (1 + zeta2) * (1 + zeta3)
    dN12 = -0.125 * (1 - zeta1) * (1 - zeta3)
    dN22 = -0.125 * (1 + zeta1) * (1 - zeta3)
    dN32 =  0.125 * (1 + zeta1) * (1 - zeta3)
    dN42 =  0.125 * (1 - zeta1) * (1 - zeta3)
    dN52 = -0.125 * (1 - zeta1) * (1 + zeta3)
    dN62 = -0.125 * (1 + zeta1) * (1 + zeta3)
    dN72 =  0.125 * (1 + zeta1) * (1 + zeta3)
    dN82 =  0.125 * (1 - zeta1) * (1 + zeta3)
    dN13 = -0.125 * (1 - zeta1) * (1 - zeta2)
    dN23 = -0.125 * (1 + zeta1) * (1 - zeta2)
    dN33 = -0.125 * (1 + zeta1) * (1 + zeta2)
    dN43 = -0.125 * (1 - zeta1) * (1 + zeta2)
    dN53 =  0.125 * (1 - zeta1) * (1 - zeta2)
    dN63 =  0.125 * (1 + zeta1) * (1 - zeta2)
    dN73 =  0.125 * (1 + zeta1) * (1 + zeta2)
    dN83 =  0.125 * (1 - zeta1) * (1 + zeta2)

    DN = np.array([
        [dN11, dN12, dN13],
        [dN21, dN22, dN23],
        [dN31, dN32, dN33],
        [dN41, dN42, dN43],
        [dN51, dN52, dN53],
        [dN61, dN62, dN63],
        [dN71, dN72, dN73],
        [dN81, dN82, dN83]
    ])

    return N, DN



def Tet4(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates N and DN for tetrahedron element with 4 nodes
    """
    if not isinstance(zeta, np.ndarray):
        raise TypeError(f"Expected zeta as np.ndarray, got {type(zeta)}.")
    if zeta.size != 3:
        raise ValueError("Needs to be three coordinates!")

    zeta1 = float(zeta[:, 0])
    zeta2 = float(zeta[:, 1])
    zeta3 = float(zeta[:, 2])

    # N
    N1 = 1.0 - zeta1 - zeta2 - zeta3
    N2 = zeta1
    N3 = zeta2
    N4 = zeta3
    N  = np.array([[N1, N2, N3, N4]])

    # DN 
    DN = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0]
    ])

    return N, DN



def Wedge6(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates N and DN for wedge element with 4 nodes
    """
    if not isinstance(zeta, np.ndarray):
        raise TypeError(f"Expected zeta as np.ndarray, got {type(zeta)}.")
    if zeta.size != 3:
        raise ValueError("Needs to be three coordinates!")

    zeta1 = float(zeta[:, 0])
    zeta2 = float(zeta[:, 1])
    zeta3 = float(zeta[:, 2])

    # N
    N1 = 0.5 * (1 - zeta3) * (1 - zeta1 - zeta2)
    N2 = 0.5 * (1 - zeta3) * zeta1
    N3 = 0.5 * (1 - zeta3) * zeta2
    N4 = 0.5 * (1 + zeta3) * (1 - zeta1 - zeta2)
    N5 = 0.5 * (1 + zeta3) * zeta1
    N6 = 0.5 * (1 + zeta3) * zeta2
    N  = np.array([[N1, N2, N3, N4, N5, N6]])

    # DN
    dN11 = -0.5 * (1 - zeta3)
    dN12 = -0.5 * (1 - zeta3)
    dN13 = -0.5 * (1 - zeta1 - zeta2)
    dN21 =  0.5 * (1 - zeta3)
    dN22 =  0.0
    dN23 = -0.5 * zeta1
    dN31 =  0.0
    dN32 =  0.5 * (1 - zeta3)
    dN33 = -0.5 * zeta2
    dN41 = -0.5 * (1 + zeta3)
    dN42 = -0.5 * (1 + zeta3)
    dN43 =  0.5 * (1 - zeta1 - zeta2)
    dN51 =  0.5 * (1 + zeta3)
    dN52 =  0.0
    dN53 =  0.5 * zeta1
    dN61 =  0.0
    dN62 =  0.5 * (1 + zeta3)
    dN63 =  0.5 * zeta2

    DN = np.array([
        [dN11, dN12, dN13],
        [dN21, dN22, dN23],
        [dN31, dN32, dN33],
        [dN41, dN42, dN43],
        [dN51, dN52, dN53],
        [dN61, dN62, dN63],
    ])

    return N, DN


def per_eletype(etype: str, z: np.ndarray, tol: float = 1e-15) -> None:
    """
    Function to calculate and check N and DN for an element. Sum of all N should be 1 and colum sum of Dn should be 0.
    """
    N, DN = ShapeFunctions(etype, z)
    sum_N = float(np.sum(N))
    sum_DN = np.sum(DN, axis=0)

    print(f"Testing Eletype:{etype}")
    print(f"z: {z}, N shape: {N.shape}, DN shape: {DN.shape}")
    print(f"Sum(N): {sum_N}")
    print(f"Column sums DN: {sum_DN}")

    if abs(sum_N - 1.0) > tol:
        print("Incorrect N")
    elif np.all(np.abs(sum_DN) > tol):
            print("Incorrect Dn")
    else:
        print("Shape functions are correct!")

    
