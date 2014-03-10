from numpy import *
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
from pylab import *



import sys
import pickle
import scipy.sparse

def decrement(x):
	return x - 1

def clean(str):
	str = str.strip("rowscolsvals: ")
	return str

def load_iri_data():
	data = file(FEATURES).read().split("\n")
	data = map(clean, data)
	labs = array(map(lambda x : 1 if x == 1 else -1, map(int, file(LABELS).read().strip().split("\n"))))
	rows = map(int, data[0].split(","))
	cols = map(int, data[1].split(","))
	vals = array(map(int, data[2].split(",")))
	rows = array(map(decrement, rows))
	cols = array(map(decrement, cols))
	n_rows = len(labs)
	n_cols = cols.max()+1
	labs = array(labs, dtype=float64)
	data = scipy.sparse.coo_matrix( (vals, (rows,cols) ), shape=(n_rows,n_cols), dtype=float64)
	data = scipy.sparse.csc_matrix(data.T)
	return (data, labs)


for D in ["go_bp", "go_cc", "go_mf", "goip", "ip"]:
	for S in ["2", "3"]:

		FEATURES="data/iri2/gisIn33_dmel08HcLcT%s-500_%s_protId_SVM.txt" % (S, D)
		LABELS="data/iri2/gisIn33_dmel08HcLcT%s-500_%s_protId_SVM_labels.txt" % (S, D)
		pkl="data/iri2/%s_%s.pkl" % (S, D)

		print "Pickling to: ", pkl
		output = open(pkl, 'wb')
		data, labs = load_iri_data()
		G = [data, labs, [], "Comments: data is scipy.sparse.csc_matrix. No group data"]
		pickle.dump(G, output)

		




