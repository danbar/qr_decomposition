"""Module for Givens rotation."""

from math import hypot

import numpy as np


def gram_schmidt_process(A):
    """Perform QR decomposition of matrix A using Gram-Schmidt process."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize empty orthogonal matrix Q.
    Q = np.empty([num_rows, num_rows])
    cnt = 0

    # Compute orthogonal matrix Q.
    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            proj = np.dot(np.dot(Q[:, i].T, a), Q[:, i])
            u -= proj

        e = u / np.linalg.norm(u)
        Q[:, cnt] = e

        cnt += 1  # Increase columns counter.

    # Compute upper triangular matrix R.
    R = np.dot(Q.T, A)

    return (Q, R)


def givens_rotation(A):
    """Perform QR decomposition of matrix A using Givens rotation."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = A

    # Iterate over lower triangular matrix.
    (rows, cols) = np.tril_indices(num_rows, -1, num_cols)
    for (row, col) in zip(rows, cols):

        # Compute Givens rotation matrix and
        # zero-out lower triangular matrix entries.
        if R[row, col] != 0:
            (c, s) = _givens_rotation_matrix_entries(R[col, col], R[row, col])

            G = np.identity(num_rows)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s

            R = np.dot(G, R)
            Q = np.dot(Q, G.T)

    return (Q, R)


def _givens_rotation_matrix_entries(a, b):
    """Compute matrix entries for Givens rotation."""
    r = hypot(a, b)
    c = a/r
    s = -b/r

    return (c, s)
