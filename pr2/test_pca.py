from lab2 import pca_project
import numpy as np

P = np.random.rand(6,6)
mu = np.random.rand(6)
dim = 4

# first scenario project single row
single_row = np.random.rand(6)
result1 = pca_project(single_row, P, mu, dim)
assert(result1.shape == (4,))

# second scenario project multiple rows
two_rows = np.array([single_row, np.random.rand(6)])
result2 = pca_project(two_rows, P, mu, dim)
assert(result2.shape == (2,4))
tmp = result2[0]
assert(np.array_equal(tmp, result1))
