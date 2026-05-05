import unittest
from src.matrix_operations import rref

class TestRref(unittest.TestCase):

    def test_non_trivial_and_invertible(self):
        # arrange
        matrix = [
            [1, 2, -1, 1, 0, 0],
            [-2, 0, 1, 0, 1, 0],
            [1, -1, 0, 0, 0, 1],
        ]
        expected = [
            [1.0, 0.0, 0.0, 1.0, 1.0, 2.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        ]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)

    def test_identity_matrix(self):
        # arrange
        matrix = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        expected = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)

    def test_already_in_rref(self):
        # arrange
        matrix = [
            [1, 0, 2, 0],
            [0, 1, -3, 0],
            [0, 0, 0, 1]
        ]
        expected = [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, -3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)

    def test_requires_row_swap(self):
        # arrange
        matrix = [
            [0, 1],
            [1, 0]
        ]
        expected = [
            [1.0, 0.0],
            [0.0, 1.0]
        ]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)

    def test_zero_row_at_bottom(self):
        # arrange
        matrix = [
            [1, 2, 3],
            [0, 0, 0],
            [0, 1, 4]
        ]
        expected = [
            [1.0, 0.0, -5.0],
            [0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0]
        ]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)

    def test_all_zero_matrix(self):
        # arrange
        matrix = [
            [0, 0],
            [0, 0]
        ]
        expected = [
            [0.0, 0.0],
            [0.0, 0.0]
        ]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)

    def test_dependent_rows(self):
        # arrange
        matrix = [
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9]
        ]
        expected = [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)

    def test_rectangular_more_columns_than_rows(self):
        # arrange
        matrix = [
            [1, 2, 3, 4],
            [2, 4, 6, 8]
        ]
        expected = [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0]
        ]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)

    def test_rectangular_more_rows_than_columns(self):
        # arrange
        matrix = [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
        expected = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0]
        ]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)

    def test_single_row(self):
        # arrange
        matrix = [[2, 4, 6]]
        expected = [[1.0, 2.0, 3.0]]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)

    def test_single_column(self):
        # arrange
        matrix = [
            [2],
            [4],
            [6]
        ]
        expected = [
            [1.0],
            [0.0],
            [0.0]
        ]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)

    def test_fractional_result(self):
        # arrange
        matrix = [
            [2, 1],
            [1, 1]
        ]
        expected = [
            [1.0, 0.0],
            [0.0, 1.0]
        ]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)

    def test_augmented_system_unique_solution(self):
        # arrange
        matrix = [
            [1, 1, 3],
            [2, -1, 0]
        ]
        expected = [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0]
        ]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)

    def test_augmented_system_inconsistent(self):
        # arrange
        matrix = [
            [1, 1, 1],
            [1, 1, 2]
        ]
        expected = [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        # act
        observed = rref(matrix)
        # assert
        self.assertEqual(observed, expected)


if __name__ == '__main__':
    unittest.main()
