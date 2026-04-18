def identity_matrix(n):
    """
    :param n:
    :return: returns an 'n by n' identity matrix.
    """
    if n < 1:
        raise ValueError("'n' must be a positive integer")

    return [[1 if i==j else 0 for j in range(n)] for i in range(n)]