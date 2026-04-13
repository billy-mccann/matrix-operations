import unittest
from matrix_operations import invert, identity_matrix, multiply_matrices

class TestInvert(unittest.TestCase):

    def test_invert_linearly_dependent_rows_raises_error(self):
        # arrange
        matrix_a = [
            [1, 1, 1],
            [1, 1, 2],
            [1, 1, 3],
        ]
        # act/assert
        with self.assertRaises(ValueError):
            invert(matrix_a)

    def test_invert_not_square_raises_error(self):
        # arrange
        matrix_a = [
            [1, 1, 1],
            [1, 1, 2],
        ]
        # act/assert
        with self.assertRaises(ValueError):
            invert(matrix_a)

    def test_invertible_matrix_succeeds(self):
        # arrange
        matrix_a = [
            [1, 2, -1],
            [-2, 0, 1],
            [1, -1, 0],
        ]
        expected = [
            [1.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [2.0, 3.0, 4.0],
        ]
        # act
        matrix_b = invert(matrix_a)
        # assert
        self.assertEqual(matrix_b, expected)
        self.assertInverseMatrices(matrix_a, matrix_b)

    def assertInverseMatrices(self, matrix_a, matrix_b):
        identity = identity_matrix(len(matrix_a))
        self.assertEqual(multiply_matrices(matrix_a, matrix_b), identity)

if __name__ == '__main__':
    unittest.main()
