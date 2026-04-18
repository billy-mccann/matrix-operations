from MLFromScratch.matrix_operations import transpose, invert, multiply_matrices

def scale_entries(x):
    xt = transpose(x)
    for row in xt:
        max_entry = max(row)
        if max_entry != 0:
            for j in range(len(row)):
                row[j] /= max_entry
    #xt = [[(entry/max(row)) for entry in row] for row in xt]
    return transpose(xt)

def clean_floating_point_errors(x, epsilon = 1.0e-10):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if abs(x[i][j]) < epsilon:
                x[i][j] = 0.0
            if abs(x[i][j] - 1) < epsilon:
                x[i][j] = 1.0
    return x

def add_ones_column(x):
    """
    This method merely adds a '1' to the front of each row for
    conveniently computing w_0

    THE RESPONSIBILITY TO PROVIDE PROPERLY FORMED INPUT LIES WITH THE CALLER
    Input should be arranged as follows:

    :param x: x needs to be an 'n x d' matrix, completely filled.

            Each row of x represents a data point of 'd' dimensions.
            For the common example:
                x[0] = (2, 2100); x[1] = (3, 2700)
            Then for any i,  x[i] = (num_bedrooms, square_feet)
            Here d = 2.

            Each column of x represents the slice of all n entries for x_*i
            e.g. x[*][0] is always 'num_bedrooms'; x[*][1] is always 'square_feet'
    :return:

        Simply appends a '1' onto each row, resulting in an 'n x d+1' matrix

        | 1 x_11 x_12 ... x_1d|     |1  --x_1^T--  |
        | 1 x_21 x_22 ... x_2d|     |1  --x_2^T--  |
        |       ...           |     |1     ...     |
    X = | 1 x_i1 x_i2 ... x_id|  =  |1  --x_i^T--  |
        |       ...           |     |1     ...     |
        | 1 x_n1 x_12 ... x_nd|     |1  --x_n^T--  |
    """
    return [[1] + row for row in x]

if __name__ == '__main__':
    x = [
        [2, 2000],
        [2, 2100],
        [2, 1800],
        [3, 2500],
        [3, 2600],
        [3, 2100],
    ]
    x = add_ones_column(x)
    x = scale_entries(x)
    print(f"x = {x}")
    xt = transpose(x)
    print(f"xt = {xt}")

    xtx = multiply_matrices(xt, x)
    print(f"xtx = {xtx}")

    xtx_inverse = invert(xtx)
    print(f"xtx_inverse = {xtx_inverse}")

    identity_ish = multiply_matrices(xtx, xtx_inverse)
    print(f"close to identity = {identity_ish}")

    identity = clean_floating_point_errors(identity_ish)
    print(f"should be identity = {identity}")

