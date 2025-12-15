"""
Written by: Anand Mathew
Shape functions testing file
"""

# Opening rituals
import numpy as np
from typing import Tuple


def test_shape_functions():
    """
    Testing each and every element
    """
    
    # 1D
    per_eletype("l2",  np.array([[0.3]]))
    per_eletype("l3",  np.array([[0.3]]))

    # 2D 
    per_eletype("q4",  np.array([[0.3, 0.3]]))
    per_eletype("q8",  np.array([[0.3, 0.3]]))
    per_eletype("t3",  np.array([[0.3, 0.3]]))
    per_eletype("t6",  np.array([[0.3, 0.3]]))

    # 3D
    per_eletype("b8",   np.array([[0.3, 0.3, 0.3]]))
    per_eletype("tet4", np.array([[0.3, 0.3, 0.3]]))
    per_eletype("w6",   np.array([[0.3, 0.3, 0.3]]))


    try:
        per_eletype("b8",   np.array([[0.3, 0.3]]))  
    except ValueError as e:
        print("PASS: Not correct zeta", e)


def main():
    """
    Main function to run all tests for the three functions.
    """
    test_shape_functions()

if __name__ == "__main__":
    main()

