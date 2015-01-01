"""Module for Givens rotation."""

from math import hypot

import numpy as np


def qr_decomposition(A):
    """Perform QR decomposition of matrix A."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = A

    # Iterate over lower triangular matrix.
    (rows, cols) = np.tril_indices(num_rows, -1, num_cols)
    for (row, col) in zip(rows, cols):

        # Compute Givens rotation matrix in order to
        # zero-out lower triangular matrix entries.
        if R[row, col] != 0:
            (c, s) = givens_rotation(R[col, col], R[row, col])

            G = np.identity(num_rows)
            G[col, col] = c
            G[row, row] = c
            G[row, col] = s
            G[col, row] = -s

            R = np.dot(G, R)
            Q = np.dot(Q, G.T)

    return (Q, R)


def givens_rotation(a, b):
    """Compute Givens rotation."""
    r = hypot(a, b)
    c = a/r
    s = -b/r

    return (c, s)
