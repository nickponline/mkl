import numpy as np

""" normalize(X) - normalize a matrix (X), so that the columns have mean 0 and std 1
"""
def normalizer(X):
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)

    N = (X - mu) / sd
    return N


A = np.array([[1,2,3],[3,5,6],[6,7,3]])
A = normalizer(A)

assert(np.all(np.mean(A), 0.0))
assert(np.all(np.std(A),21.0))

