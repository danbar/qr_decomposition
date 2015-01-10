qr_decomposition
================

qr_decomposition is a Python package for computing the QR decomposition.

Therefore, the following algorithms are implemented:
* Gram-Schmidt process
* Householder reflection
* Givens rotation

Dependencies
------------

* Python 3.4 or later
* NumPy 1.9 or later

Example
-------

A simple example of how to use the qr_decomposition package.

```Python
import numpy as np

from qr_decomposition import qr_decomposition

# Set print options (optional)
np.set_printoptions(precision=4, suppress=True)

# Input matrix
A = np.array([[6, 5, 0],
              [5, 1, 4],
              [0, 4, 3]])

# Print input matrix
print(A)

# Compute QR decomposition using Givens rotation
(Q, R) = qr_decomposition.givens_rotation(A)

# Print orthogonal matrix Q
print(Q)

# Print upper triangular matrix R
print(R)
```
