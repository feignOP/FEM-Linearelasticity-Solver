
"""
Written by: Anand Mathew
"""

# Opening Rituals
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss


def test_GaussPoints():
    """
    Basic unit tests for GaussPoints function.

    Checks
    ----------
    - 1D, 2D (q4, t3), and 3D (b8, tet4) total weights.
    - Invalid dimension, element type, and NGPTS handling.
    """ 
    
    print("Running basic GaussPoints tests.\n")

    # 1D check
    r, w = GaussPoints(1, "ignored", 8)
    print(f"1D expected: 2.0, got: {sum(w)}")
    print(f"1D expected sum of r: 0.0, got: {sum(r)}")

    # 2D q4 check
    r, w = GaussPoints(2, "q4", 8)
    print(f"2D Q4 expected: 4.0, got: {sum(w)}")
    print(f"Expected sum of rx: 0.0, got: {sum(r[:, 0])} and ry: 0.0, got: {sum(r[:, 1])}")

    # 3D b8 check
    r, w = GaussPoints(3, "b8", 2)
    print(f"3D B8 expected: 8.0, got: {sum(w)}")

    # 2D t3 check
    r, w = GaussPoints(2, "t3", 1)
    print(f"2D T3 expected: 0.5, got: {sum(w)}")

    # 3D tet4 check
    r, w = GaussPoints(3, "tet4", 1)
    print(f"3D TET4 expected: {1.0/6.0}, got: {sum(w)}")

    # Invalid dimension
    try:
        GaussPoints(0, "q4", 2)
    except Exception as e:
        print(f"Invalid dimension test raised error: {e}")

    # Invalid element
    try:
        GaussPoints(2, "Nakshatrala", 2)
    except Exception as e:
        print(f"Invalid element type test raised error: {e}")

    # Invalid NGPTS
    try:
        GaussPoints(1, "q4", "a")
    except Exception as e:
        print(f"Invalid NGPTS test raised error: {e}")

    print("\nAll basic tests completed.")


if __name__ == "__main__":
    # Test function
    test_GaussPoints()


    # Docstring test
    print("Printing docstring for 3d")
    help (Gauss_3D)