from numpy import *
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
from pylab import *
from kernel_learning import *
from scipy.sparse import vstack, hstack
import django
from django.template import Template, Context
from shogun.Mathematics import Math_init_random
from collections import namedtuple
from runsetup import *
import csv
import shelve
import datetime

import sys
import pickle
import scipy.sparse
import numpy

store = shelve.open("results.shelve")

for i, key in enumerate(store.keys()):
	print "RUN: ",i, key
	run = store[key] 
	print "Classifier:", run.classifiertype
	print "Modelselection:", run.modelselection
	print "Dataset:", run.dataset
	results = run.results
	for result in results:
		print result.metric, result.mean

	print ""