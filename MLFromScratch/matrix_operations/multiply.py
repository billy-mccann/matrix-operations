"""
 Multiply matrices A*B = C
"""

def multiply_matrices(a, b):
    """
    :param a: A, a matrix of size 'm by k'
    :param b: B, a matrix of size 'k by n'
    :return: C, a matrix of size 'm by n'

    multiply_matrices() accepts matrices of any size, but raises a
    ValueError if they are not properly sized to be multiplied.
    """

    if not a or not b or len(a) < 1 or len(b) < 1:
        raise ValueError("Matrices must not be empty.")

    (m, n, k) = (len(a), len(b[0]), len(a[0]))

    if len(b) != k:
        raise ValueError("Matrices are the wrong size.")

    return [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(n)] for i in range(m)]
