import pickle
import numpy as np

for s in ['hia', 'pgp', 'tdp']:
	path = "data/xue/%s.pkl" % s
	pkl = open(path, "r")
	R, y, g, notes = pickle.load(pkl)

	print np.array(g).shape
	print R.shape
	R = R

	for t in xrange(1, 6):
		idx = []
		for i,v in enumerate(g[0]):
			if (int(v) == t):
				idx.append(i)

		print idx

		X = R[idx,:]
		data = [X,y,None, "Comments: data matrix is dense, subgroup %d" %t]
		output = open("data/xue/%s.%d.pkl" % (s,t), 'wb')
		print "Saving: " "data/xue/%s.%d.pkl" % (s,t)
		pickle.dump(data, output)