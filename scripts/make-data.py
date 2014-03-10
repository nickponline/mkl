import numpy as np
import scipy as sp
import pickle

def pickle_dataset(name):
	path = "%s.csv" % name
	pkl  = "%s.pkl" % name
	data = open(path).read().strip()

	X = []
	y = []
	g = []

	for i, row in enumerate(data.split("\n")):
		#print row
		if i == 0:
			col = row.split(",")
			g.append(map(float, col[2::]))
		if i > 1:
			col = row.split(",")
			y.append(int(col[0]))
			X.append(map(float, col[2::]))

	X = np.array(X)
	y = np.array(y)
	g = np.array(g)

	data = [X,y,g, name]
	output = open(pkl, 'wb')
	pickle.dump(data, output)

	#print X.shape
	#print y.shape
	#print g.shape

	print "Finished: ", name


map(pickle_dataset, ["hia", "pgp", "tdp"])


