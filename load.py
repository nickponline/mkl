import scipy.sparse 

from scipy import *
from numpy import *
from pylab import *

#from shogun import *

def decrement(x):
	return x - 1

def load_iri_data():
	data = file("data/iri/gisInv30-130_da_SVM.txt").read().split("\n")
	labs = array(map(lambda x : 1 if x == 1 else -1, map(int, file("data/iri/gisInv30-130_da_SVM_l.txt").read().split("\n"))))
	rows = map(int, data[0].split(","))
	cols = map(int, data[1].split(","))
	vals = array(map(int, data[2].split(",")))
	rows = array(map(decrement, rows))
	cols = array(map(decrement, cols))
	n_rows = len(labs)
	n_cols = cols.max()+1
	data = scipy.sparse.coo_matrix( (vals, (rows,cols) ), shape=(n_rows,n_cols)).todense()
	print data.shape
	print labs.shape
	return (data, labs)

print "Loading"
a, b = load_iri_data()
print a.T.shape