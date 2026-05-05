def transpose(a):
    """
    :param a: matrix, intended to be used with [[float]] or [[int]]
    :return: transpose of matrix a: a_ij ==> a_ji
    """
    if not a or len(a) < 1 or len(a[0]) < 1:
        raise ValueError("Matrix must not be empty.")
    (m, n) = (len(a), len(a[0]))

    return [[a[i][j] for i in range(m)] for j in range(n)]