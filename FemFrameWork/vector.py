"""
Written by: Anand Mathew
"""


# Opening rituals
import numpy as np
from typing import Tuple

def Vector(A: np.ndarray) -> np.ndarray:
    """
    Stacks the columns of a matrix A into a single column vector.

    Parameters
    ----------
    A : np.ndarray
        Input matrix of size (m, n)

    Returns
    -------
    vecA : np.ndarray
        Column vector (m*n, 1) containing elements of A stacked column by column.
    """

    if not isinstance(A, np.ndarray):
        raise TypeError(f"Expected input of type np.ndarray, got {type(A)} instead.")

    if A.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, but got an array with ndim={A.ndim}.")

    vec = A.flatten(order='F').reshape(-1, 1)

    return vec

