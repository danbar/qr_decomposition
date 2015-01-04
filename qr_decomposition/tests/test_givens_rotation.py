"""
Python unit-test
"""

import unittest

import numpy as np
import numpy.testing as npt

from .. import qr_decomposition


class TestGivensRotation(unittest.TestCase):
    """Test case for QR decomposition using Givens rotation."""

    def test_wikipedia_example1(self):
        """Test of Wikipedia example

        The example for the following QR decomposition is taken from
        https://en.wikipedia.org/wiki/Givens_rotation#Triangularization.
        """

        A = np.array([[6, 5, 0],
                      [5, 1, 4],
                      [0, 4, 3]], dtype=np.float64)

        (Q, R) = qr_decomposition.givens_rotation(A)

        Q_desired = np.array([[0.7682, 0.3327, 0.5470],
                              [0.6402, -0.3992, -0.6564],
                              [0, 0.8544, -0.5196]], dtype=np.float64)
        R_desired = np.array([[7.8102, 4.4813, 2.5607],
                              [0, 4.6817, 0.9664],
                              [0, 0, -4.1843]], dtype=np.float64)

        npt.assert_almost_equal(Q, Q_desired, 4)
        npt.assert_almost_equal(R, R_desired, 4)

    def test_wikipedia_example2(self):
        """Test of Wikipedia example

        The example for the following QR decomposition is taken from
        http://de.wikipedia.org/wiki/Givens-Rotation.
        """

        A = np.array([[3, 5],
                      [0, 2],
                      [0, 0],
                      [4, 5]], dtype=np.float64)

        (Q, R) = qr_decomposition.givens_rotation(A)

        Q_desired = np.array([[0.6, 0.3577, 0, -0.7155],
                              [0, 0.8944, 0, 0.4472],
                              [0, 0, 1, 0],
                              [0.8, -0.2683, 0, 0.5366]], dtype=np.float64)
        R_desired = np.array([[5, 7],
                              [0, 2.2360],
                              [0, 0],
                              [0, 0]], dtype=np.float64)

        npt.assert_almost_equal(Q, Q_desired, 4)
        npt.assert_almost_equal(R, R_desired, 4)


if __name__ == "__main__":
    unittest.main()
