import numpy as np

def make_training_test(data, labs):
	
	rows, cols = data[0].shape
	ratio = np.floor(cols * 0.8)
	perm = np.random.permutation(cols)

	training_idx = perm[0:ratio]
	testing_idx = perm[ratio:]

	print training_idx
	print testing_idx

	training_data, testing_data = [], []
	training_labs, testing_labs = None, None

	for d in data:
		training_data.append(d[:, training_idx])
		testing_data.append(d[:, testing_idx])

		training_labs = labs[training_idx]
		testing_labs = labs[testing_idx]

	return training_data, testing_data, training_labs, testing_labs


X = np.random.rand(5, 5)
y = np.array([xrange(5)]).T

print X
print y

a, b, c, d = make_training_test([X], y)

print a[0]
print b[0]
print c
print d
