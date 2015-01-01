"""Example for the usage of the qr_composition package."""
import numpy as np

from qr_decomposition import qr_decomposition

# Set print options (optional)
np.set_printoptions(precision=4, suppress=True)

# Input matrix
# A = np.array([[6, 5, 0], [5, 1, 4], [0, 4, 3]]).
A = np.array([[3, 5],
              [0, 2],
              [0, 0],
              [4, 5]])

# Print input matrix
print(A)

# Compute QR decomposition using Givens rotation
(Q, R) = qr_decomposition.givens_rotation(A)

# Print orthogonal matrix Q
print(Q)

# upper triangular matrix R
print(R)
