import numpy as np
import pylab as pl
import random
import scipy.io

files = ["/Users/nickp/Dropbox/Repos/mkl/matlab/hia.txt.0.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/hia.txt.1.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/hia.txt.2.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/hia.txt.3.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/hia.txt.4.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/pgp.txt.0.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/pgp.txt.1.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/pgp.txt.2.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/pgp.txt.3.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/pgp.txt.4.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/tdp.txt.0.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/tdp.txt.1.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/tdp.txt.2.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/tdp.txt.3.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/tdp.txt.4.mat.sav", "/Users/nickp/Dropbox/Repos/mkl/matlab/synthredundant.txt.0.mat.sav"]
#files = [files[-1]]

for i, f in enumerate(files):
	mat = []
	spr = []
	raw = open(f).read().strip().split("\n")
	raw = filter(lambda h: len(h) > 0, raw)
	raw = filter(lambda h: h[0] != '#', raw)

	for r in xrange(len(raw)-1):
		mat.append(map(float, raw[r].strip().split(" ")))
	spr = map(float, raw[len(raw)-1].strip().split(" "))

	mat = np.array(mat)
	spr = np.array(spr)

	# for k in xrange(100):
	# 	for j in xrange(20):
	# 		mat[j, k]= 1.0 - random.random() / 5.0
	

	# for k in [23, 43, 65, 34, 78, 13, 1, 45, 67, 94]:
	# 	for j in xrange(20):
	# 		mat[j, k]= random.random() / 10.0
	
	N = mat.shape[1]

	pl.figure()
	pl.bar(np.arange(1, N+1), np.mean(mat, axis=0), 0.4, color='r', yerr=np.std(mat, axis=0) / 2.0)
	pl.ylim(0, np.mean(mat + np.std(mat, axis=0) / 2.0, axis=0).max())
	pl.xlim(1, N+1)
	pl.xlabel("Feature")
	pl.ylabel("Unimportance")
	pl.savefig("/Users/nickp/Dropbox/PhD/Thesis/img/ardgp-%d.eps" % i)
	pl.show()
	print mat.shape

	
