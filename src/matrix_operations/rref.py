def rref(input_matrix, epsilon: float = 1e-10) -> list[list[float]]:
    """

    RREF (Reduced Row Echelon Form) computation.

    Implements Gaussian elimination with optional floating-point tolerance.
    Supports rectangular matrices and returns a new matrix in RREF.

    Assumes input is a list of lists of numbers.
 
    :param input_matrix: the matrix to be row-reduced.
    :param epsilon: used for conversion to zero in floating point arithmetic.
    :return: input_matrix in its row-reduced-echelon-form, represented as floats.

    rref() converts values to floats prior to row-reduction. As such, it works
    with floats or ints.
    """

    if not input_matrix:
        raise ValueError("Matrix must not be empty.")

    num_rows = len(input_matrix)
    num_cols = len(input_matrix[0])

    for row in input_matrix:
        if len(row) != num_cols:
            raise ValueError("Oops! All rows must have the same length!")

    # Convert to floats to avoid integer division
    result = [list(map(float, row)) for row in input_matrix]

    #Begin the actual algorithm
    pivot_row = 0
    for pivot_col in range(num_cols):
        if pivot_row >= num_rows:
            break

        # Find pivot
        row_to_swap = None
        for row in range(pivot_row, num_rows):
            if abs(result[row][pivot_col]) > epsilon:
                row_to_swap = row
                break
        if row_to_swap is None:
            continue
        result[pivot_row], result[row_to_swap] = result[row_to_swap], result[pivot_row]

        # Normalize pivot row
        pivot = result[pivot_row][pivot_col]
        result[pivot_row] = [x / pivot for x in result[pivot_row]]

        # Eliminate entries in pivot column of other rows
        for row in range(num_rows):
            if row != pivot_row and abs(result[row][pivot_col]) > epsilon:
                factor = result[row][pivot_col]
                result[row] = [
                    result[row][col] - factor * result[pivot_row][col]
                    for col in range(num_cols)
                ]
        pivot_row += 1

    # Clean floating point errors.
    for i in range(num_rows):
        for j in range(num_cols):
            if abs(result[i][j]) < epsilon:
                result[i][j] = 0.0

    return result
