from time import clock, time
import cma
import csv
import shelve
import datetime
import random
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

evals = 0




def save(object, filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
    """
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()

def load(filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
    """
    file = gzip.GzipFile(filename, 'rb')
    result = cPickle.load(file)
    file.close()
    return result

def make_bounds(parameters, bound):
    
    return [(np.power(2.0,-float(bound)), np.power(2.0,float(bound))) for _ in xrange(int(parameters)+2)]


class Result():
    def __init(self):
        mean = 0
        conf_int_up = 0

def objective(parameters, metrics = [ACCURACY], production=False):
    global evals
    evals += 1
    Math_init_random(252)

    
    parameters = map(abs, map(float, list(parameters)))
    for i in xrange(len(parameters)):
        if parameters[i] == 0.0:
            parameters[i] = 1e-10
    
    c  = parameters[0]
    w  = parameters[1:]

    
    set_kernel_parameters(kernel, w, PARAMETERS['KERNEL'])
    
    
    classifier.set_C(c, c)
    
    splitting_strategy   = StratifiedCrossValidationSplitting(train_labels, int(PARAMETERS['FOLDS']))

    # splitting_strategy.build_subsets()
    # subs = splitting_strategy.get_num_subsets()
    # for i in xrange(subs):
    #     train_idx = splitting_strategy.generate_subset_indices(i)
    #     test_idx = splitting_strategy.generate_subset_inverse(i)
    #     train_data = train

    result = None
    for metric in metrics:
        evaluation_criterium = ContingencyTableEvaluation(metric)
        cross_validation     = CrossValidation(classifier, train_features, train_labels, splitting_strategy, evaluation_criterium)
            
        cross_validation.set_num_runs(int(PARAMETERS['RUNS']))
        cross_validation.set_conf_int_alpha(0.05)
        
        #cross_validation.set_autolock(False)
        result = Result()
        try:
            result = None
            result = cross_validation.evaluate()
        except:
            result = Result()
            result.mean = 0
            result.conf_int_up = 0

        fmt_parameters = map(lambda h: "%3.5f" % h, parameters)
        if production:
            print "%2.3f$\\pm$%2.3f &" % (result.mean, result.conf_int_up-result.mean), 
        else:
            #pass
            print "Objective & %2.5f +/- %2.5f &" % (result.mean, result.conf_int_up-result.mean), "Args => ",fmt_parameters, "Metric => ", metric
    return result.mean

def min_objective(parameters):
    return -objective(parameters)

def normalizer(X):
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    
    N = (X - mu) / sd
    return N


def auto_convolve(X):
    rows,cols = X.shape
    N = np.zeros((rows * rows, cols))

    for lv1,i in enumerate(X):
        for lv2,j in enumerate(X):
            N[lv1 * rows + lv2, :] = (i*j)

    return N

def convolve_features(data, K, ctype):
    N = len(data)
    for i in xrange(N):
        t = auto_convolve(data[i])
        data.append( t )
    K = K * 2
    return data, K, "MKL"

def gauss_southwell_search(obj):
    x0 = [1.0] * (2 + int(PARAMETERS['PARAMETERS']))
    history = {}
    best_parameter = gauss_southwell(obj, x0, MAXITER=3)
    return best_parameter, history


startTime = datetime.datetime.now()


PARAMETERS                                                = load_configuration(sys.argv[1])
data, labs, ctype, K                                      = load_data(PARAMETERS['DATA'])


for i in xrange(len(data)):
	X = data[i]
	y = labs
	mat = {}
	mat['X'] = numpy.array(X)
	mat['y'] = numpy.array(y)
	scipy.io.savemat(sys.argv[1] + ".%d.mat" % i, mat)

