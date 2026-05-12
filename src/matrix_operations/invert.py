from .rref import rref
from .utilities import identity_matrix

"""
Inverse matrix computation.

The input matrix is first augmented with the identity matrix of the same dimension.
Row operations are then used to obtain the *reduced-row-echelon-form* (rref) of the 
augmented matrix.  

Supports invertible matrices only.

Assumes input is a list of lists of numbers (floats or ints).
"""

def invert(input_matrix) -> list[list[float]]:
    """
    :param input_matrix: a matrix of ints or floats
    :return: a matrix of floats

    invert() raises a ValueError when fed a singular (i.e. non-invertible) matrix.
    """
    if not input_matrix:
        raise ValueError("Matrix must not be empty.")

    num_rows = len(input_matrix)
    num_cols = len(input_matrix[0])

    if num_cols != num_rows:
        raise ValueError("Matrix must be square")

    identity = identity_matrix(num_rows)
    augmented = [input_matrix[i] + identity[i] for i in range(num_rows)]
    row_reduced = rref(augmented)

    # Raise error if original matrix has not reduced to the identity.
    identity_if_invertible = [row[:num_cols] for row in row_reduced]
    if identity_if_invertible != identity:
        raise ValueError("Matrix must be invertible")

    return [row[num_cols:] for row in row_reduced]
