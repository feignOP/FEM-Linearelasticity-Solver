
import numpy as np
from typing import Tuple


def test_Vector():

    # Square matrix
    A = np.array([[1, 2],
                  [3, 4]])
    expected_vec = np.array([[1], [3], [2], [4]])
    assert np.array_equal(Vector(A), expected_vec), "Column-major stacking order incorrect"

    # Rectangular matrix 
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    expected_vec = np.array([[1], [4], [2], [5], [3], [6]])
    assert np.array_equal(Vector(A), expected_vec), "Column-major stacking order incorrect"
    
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    expected_vec = np.array([[1], [3], [5], [2], [4], [6]])
    assert np.array_equal(Vector(A), expected_vec), "Column-major stacking order incorrect"

    # Row vector
    A = np.array([[1, 2, 3, 4]])
    expected_vec = np.array([[1], [2], [3], [4]])
    assert np.array_equal(Vector(A), expected_vec), "Column-major stacking order incorrect"

    # Column vector
    A = np.array([[1], [2], [3], [4]])
    expected_vec = np.array([[1], [2], [3], [4]])
    assert np.array_equal(Vector(A), expected_vec), "Column-major stacking order incorrect"

    # ! element matrix
    A = np.array([[42]])
    expected_vec = np.array([[42]])
    assert np.array_equal(Vector(A), expected_vec), "Column-major stacking order incorrect"


    # Invalid input
    try:
        Vector([1, 2, 3])  # Not an array
    except TypeError as e:
        print("PASS: Detected a not an array", e)

    try:
        Vector(np.array([1, 2, 3]))  # Not a 2D array
    except ValueError as e:
        print("PASS: tected not 2D matrix", e)

    print("All Vector tests passed successfully.")



def main():
    """
    Main function to run all tests for the three functions.
    """

    test_Vector()


if __name__ == "__main__":
    main()

