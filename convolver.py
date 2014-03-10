import numpy as np



def auto_convolve(X):
	rows,cols = X.shape
	N = np.zeros((rows * rows, cols))

	for lv1,i in enumerate(X):
		for lv2,j in enumerate(X):
			N[lv1 * rows + lv2, :] = (i*j)

	return N



A = np.array([[1,2], [3,4], [3,5]])
M = auto_convolve(A)
print A
print M

print [1] * 10