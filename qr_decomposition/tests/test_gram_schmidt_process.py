"""
Python unit-test
"""

import unittest

import numpy as np
import numpy.testing as npt

from .. import qr_decomposition


class TestGramSchmidtProcess(unittest.TestCase):
    """Test case for QR decomposition using Gram-Schmidt process."""

    def test_wikipedia_example1(self):
        """Test of Wikipedia example

        The example for the following QR decomposition is taken from
        http://en.wikipedia.org/wiki/QR_decomposition#Using_the_Gram.E2.80.93Schmidt_process
        """

        A = np.array([[12, -51, 4],
                      [6, 167, -68],
                      [-4, 24, -41]], dtype=np.float64)

        (Q, R) = qr_decomposition.gram_schmidt_process(A)

        Q_desired = np.array([[0.8571, -0.3943, -0.3314],
                              [0.4286, 0.9029, 0.0343],
                              [-0.2857, 0.1714, -0.9429]], dtype=np.float64)
        R_desired = np.array([[14, 21, -14],
                              [0, 175, -70],
                              [0, 0, 35]], dtype=np.float64)

        npt.assert_almost_equal(Q, Q_desired, 4)
        npt.assert_almost_equal(R, R_desired, 4)


if __name__ == "__main__":
    unittest.main()
