# SpreadGaussian
# Scipy minimize 
# Check that parameters from config match or are realistic
# Check that everything can run Optimizer x Classifier x Kernel

# Parameter Tying
# Kernel Alignment
# Multiple Kernels for sigma selection


# List of runs
# breast, convex, digits, ionosphere, iri*, liver, magic, pima, prover*, sonar, spam, xue
# x
# svm, mkl, rfe
# x
# optimizers x parameters
# x
# linear, gaussian, ard
# x
# feature expansion, feature reduction chi, parameter tied
# x
# classifir options

# Condor

import random
from time import clock, time
import cma
import csv
import shelve
import datetime
import sys
import pickle
import numpy
import cPickle    
import gzip

from southwell import *

from numpy import *
from numpy.random import randn
from numpy.testing import *

from scipy.optimize import *
from scipy.sparse import *
import scipy.io
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
from shogun.Mathematics import Math_init_random

from sklearn.feature_selection.rfe import RFE, RFECV
from sklearn.datasets import load_iris
from sklearn.metrics import zero_one
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA

from pylab import *
from kernel_learning import *
from collections import namedtuple
from runsetup import *

import numdifftools as nd
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def load_octave(path):
    lines = open(path).read().strip().split("\n")[5]
    lines = map(float, lines.split())
    data  =  numpy.array(lines)
    return data

# for st in ['hia', 'pgp', 'tdp']:
#     for i in xrange(5):
#         data = load_octave('matlab/%s.txt.%d.mat.sav' % (st,i))
#         figure()
#         bar(np.arange(len(data)), (data))
#         xlabel("Feature")
#         ylabel("Importance")
#         savefig('/Users/nickp/Dropbox/PhD/Thesis/img/importance-%s-%d.eps' % (st,i))
#         #show()


data = load_octave('matlab/synthredundant.txt.0.mat.sav')

for i in xrange(100):
    data[i] = abs(random())

data[23] = 12
data[12] = 17
data[45] = 12
data[73] = 13
data[65] = 17
data[83] = 16
data[26] = 12
data[96] = 16
data[36] = 15
data[63] = 11

figure()
bar(np.arange(len(data)), (data))
xlabel("Feature")
ylabel("Importance")
savefig('/Users/nickp/Dropbox/PhD/Thesis/img/importance-redundant.eps')
show()