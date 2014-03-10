from collections import namedtuple
import csv
import shelve

import sys
import pickle
import scipy.sparse
import numpy
import datetime

from kernel_learning import *

def make_runs():
	ret = []
	for dataset_name, dataset in [("dataset 1", "2"), ("dataset 2", "3")]:
		for features in [["ip", "go_bp", "go_cc", "go_mf", "goip"]]:
		#for features in [["ip"], ["go_bp"], ["go_cc"], ["go_mf"], ["goip"], ["all"], ["ip", "go_bp", "go_cc", "go_mf", "goip"]]:
			name = dataset_name + " " + " ".join(features)  + " features"
			paths = ["data/iri2/%s_%s.pkl" % (dataset, f) for f in features]
			ret.append((name, paths))
	return ret

#print datetime.datetime.now()
#print make_runs()
