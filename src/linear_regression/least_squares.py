from src.matrix_operations import transpose, invert, multiply_matrices
from tests.test_data.housing_dummy_data import DummyData

def extract_x_y(training_data: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    data_t = transpose(training_data)

    x_t = data_t[:-1]
    x = transpose(x_t)
    x = add_ones_column(x)

    y_t = data_t[-1:]
    y = transpose(y_t)
    return x, y

# useful for checking if something is close to identity. Should probably
# generalize this to all numbers.
def clean_floating_point_errors(x, epsilon = 1.0e-10):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if abs(x[i][j]) < epsilon:
                x[i][j] = 0.0
            if abs(x[i][j] - 1) < epsilon:
                x[i][j] = 1.0
    return x

def add_ones_column(matrix_x: list[list[float]]) -> list[list[float]]:
    """
    This method merely adds a '1' to the front of each row for
    conveniently computing w_0

    THE RESPONSIBILITY TO PROVIDE PROPERLY FORMED INPUT LIES WITH THE CALLER
    Input should be arranged as follows:

    :param matrix_x: x needs to be an 'n x d' matrix, completely filled.

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
    return [[1] + row for row in matrix_x]

class LinearRegression:

    def train_least_squares(training_data: list[list[float]]):
        x, y = extract_x_y(training_data)
        xt = transpose(x)
        xtx = multiply_matrices(xt, x)
        xtx_inverse = invert(xtx)
        xty = multiply_matrices(xt, y)
        w_least_squares = multiply_matrices(xtx_inverse, xty)
        w = transpose(w_least_squares)[0]

        def predict(input):
            input = [1] + input
            total = 0
            nonlocal w
            for i in range(len(input)):
                total += input[i]*w[i]
            return total

        return LRModel(predict)

class LRModel:

    def __init__(self, predict):
        self.predict = predict

if __name__ == '__main__':

    data = DummyData.housing_data
    model = LinearRegression.train_least_squares(data)

    x_test = [[1400.0, 2.0], [1732, 3], [1800, 3], [2100, 4]]

    print("About to model.....")
    for house in x_test:
        print(model.predict(house))
