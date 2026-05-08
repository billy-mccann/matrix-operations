# from typing_extensions import Doc
import src.matrix_operations as mo
from tests.test_data.housing_dummy_data import DummyData

class LRModel:
    def __init__(self, predict):
        self.predict = predict


class LinearRegression:

    @staticmethod
    def train_least_squares(training_data: list[list[float]]) -> LRModel:
        x, y = mo.extract_x_y(training_data)
        xt = mo.transpose(x)
        xtx = mo.multiply_matrices(xt, x)
        xtx_inverse = mo.invert(xtx)
        xty = mo.multiply_matrices(xt, y)
        w_ls_transpose = mo.multiply_matrices(xtx_inverse, xty)
        w_ls = mo.transpose(w_ls_transpose)[0]

        def predict(input):
            input = [1] + input
            total = 0
            nonlocal w_ls
            for i in range(len(input)):
                total += input[i]*w_ls[i]
            return total

        return LRModel(predict)


if __name__ == '__main__':

    data = DummyData.housing_data
    model = LinearRegression.train_least_squares(data)

    x_test = [[1400.0, 2.0], [1732, 3], [1800, 3], [2100, 4]]

    for house in x_test:
        print(round(model.predict(house)))

