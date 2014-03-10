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
    
 
    try:
        A = train_labels.get_labels()
    except:
        return 0.0

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
            pass
            #print "Objective & %2.5f +/- %2.5f &" % (result.mean, result.conf_int_up-result.mean), "Args => ",fmt_parameters, "Metric => ", metric
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

# Export for MATLAB analysis.
# X = data[0]
# y = labs
# mat = {}
# mat['X'] = numpy.array(X)
# mat['y'] = numpy.array(y)
# scipy.io.savemat('test.mat',mat)

#data, K, ctype                                           = convolve_features(data, K, ctype)
try:
    data                                                      = map(normalizer, data)
except:
    print "Cannot normalize sparse data!"

print "Kernels: ", K
# data, labs, ctype, K                                    = make_data(5, 200, 500, 2)
# data                                                    = make_feature_select(data, labs, FEATURES)
#training_data, testing_data, training_labs, testing_labs  = make_training_test(data, labs)
training_data = data
training_labs = labs
print "LABELS: ", labs
train_features                                            = make_features(data, PARAMETERS['KERNEL']);
train_labels                                              = make_labels(labs.T);

# test_features                                           = make_features(testing_data);
# test_labels                                             = make_labels(testing_labs);

kernel                                                    = make_kernel(K, train_features, PARAMETERS['KERNEL'], ctype=ctype, widths=[1.0] * K)
classifier                                                = make_classifier(train_features, train_labels, kernel, ctype=ctype)


describe_kernel(kernel)

history = {}
#print gaussian_process_surrogate_2d(objective)


if PARAMETERS['OPTIMIZER'] == "SurrogateSearch":
    dimensions = 2 + int(PARAMETERS['PARAMETERS'])
    
    best_parameters = None
    best_objective = None
    startTime = time()
    for idx in xrange(128):

        best_parameters = gaussian_process_surrogate_nd(objective, dimensions, search='ei')
        endTime = time()
        history = {}
        objective(best_parameters, [0, 20, 50, 40], production=True)
        print idx+1, " & ", idx+1, " & ", int(endTime) - int(startTime), "\\\\"
        results = {}
    #results = {}
    # try:
    #results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    # except:
    #     results = {}
    #     resuls["Exception"] = "Exception"

    print results
    print ""


if PARAMETERS['OPTIMIZER'] == "CG":
    PARAM_BOUND = int(PARAMETERS['PARAM_BOUND'])
    lookup = [x for x in xrange( -int(PARAM_BOUND), int(PARAM_BOUND))]
    
    for idx in xrange(5):
        
        p = [ choice(lookup) for x in xrange(2 + int(PARAMETERS['PARAMETERS']))]
        current_parameters = map( lambda h: 2 ** float(h), p)
        print "Starting from ", current_parameters
        best_parameter = None

        try:
            best_parameter = minimize(min_objective, current_parameters, method="CG")
            best_parameter = best_parameter.x
            print "Success with optimizer: ", optimizer
        except:
            print "Exception with optimizer: ", optimizer
            print sys.exc_info()[0]

    print "Best parameters: ", best_parameter
    history = {}
    objective(best_parameter)
    results = {}
    
    try:
      
        results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    except:
        results = {}
        results["Exception"] = "Exception"

    print results
    print ""

if PARAMETERS['OPTIMIZER'] == "BFGS":
    PARAM_BOUND = int(PARAMETERS['PARAM_BOUND'])
    lookup = [x for x in xrange( -int(PARAM_BOUND), int(PARAM_BOUND))]
    startTime = time()
    best = 10000
    best_parameter = None
        
    for idx in xrange(32):
        d = int(1 + int(PARAMETERS['PARAMETERS']))
        p = [ choice(lookup) for x in xrange(1 + int(PARAMETERS['PARAMETERS']))]
        current_parameters = map( lambda h: 2 ** float(h), p)
        #print "Starting from ", current_parameters
        bounds = [(2 ** -10, 2 ** 10) for _ in xrange(d)]
        jac = nd.Gradient(min_objective)
        hess = nd.Hessian(min_objective)
        parameter = minimize(min_objective, current_parameters, method="BFGS", bounds=bounds, jac=jac, hess=hess)
        if parameter.fun < best:
            best = parameter.fun
            best_parameter = parameter

        if is_power2(idx+1):  
            #print "RESULT AFTER ", idx+1                  
            endTime = time()
            #print "Evaluations: ", evals
            #print "Time Taken: ", int(endTime) - int(startTime)
            #print "Best parametsr: ", best_parameter
            history = {}
            objective(best_parameter.x, [0, 20, 50, 40], production=True)
            print idx+1, " & ", evals, " & ", int(endTime) - int(startTime), "\\\\"
            results = {}
    
    try:
      
        results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    except:
        results = {}
        results["Exception"] = "Exception"

    print results
    print ""

if PARAMETERS['OPTIMIZER'] == "NelderMead":
    PARAM_BOUND = int(PARAMETERS['PARAM_BOUND'])
    lookup = [x for x in xrange( -int(PARAM_BOUND), int(PARAM_BOUND))]
    
    for idx in xrange(100):

        p = [ choice(lookup) for x in xrange(1 + int(PARAMETERS['PARAMETERS']))]
        current_parameters = map( lambda h: 2 ** float(h), p)
        print "Starting from ", current_parameters
        best_parameter = None

        best_parameter = minimize(min_objective, current_parameters, method="Nelder-Mead", bounds=[(0.001, 1024), (0.001, 1024), (0.001, 1024)])
        print best_parameter


    print "Best parameters: ", best_parameter
    history = {}
    objective(best_parameter)
    results = {}
    
    try:
      
        results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    except:
        results = {}
        results["Exception"] = "Exception"

    print results
    print ""


if PARAMETERS['OPTIMIZER'] == "TNC":
    PARAM_BOUND = int(PARAMETERS['PARAM_BOUND'])
    lookup = [x for x in xrange( -int(PARAM_BOUND), int(PARAM_BOUND))]
    
    for idx in xrange(5):
        
        p = [ choice(lookup) for x in xrange(2 + int(PARAMETERS['PARAMETERS']))]
        current_parameters = map( lambda h: 2 ** float(h), p)
        print "Starting from ", current_parameters
        best_parameter = None

        try:
            best_parameter = minimize(min_objective, current_parameters, method="TNC")
            best_parameter = best_parameter.x
            print "Success with optimizer: ", optimizer
        except:
            print "Exception with optimizer: ", optimizer
            print sys.exc_info()[0]

    print "Best parameters: ", best_parameter
    history = {}
    objective(best_parameter)
    results = {}
    
    try:
      
        results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    except:
        results = {}
        results["Exception"] = "Exception"

    print results
    print ""
if PARAMETERS['OPTIMIZER'] == "COBYLA":
    PARAM_BOUND = int(PARAMETERS['PARAM_BOUND'])
    lookup = [x for x in xrange( -int(PARAM_BOUND), int(PARAM_BOUND))]
    
    for idx in xrange(5):
        
        p = [ choice(lookup) for x in xrange(2 + int(PARAMETERS['PARAMETERS']))]
        current_parameters = map( lambda h: 2 ** float(h), p)
        print "Starting from ", current_parameters
        best_parameter = None

        try:
            best_parameter = minimize(min_objective, current_parameters, method="COBYLA")
            best_parameter = best_parameter.x
            print "Success with optimizer: ", optimizer
        except:
            print "Exception with optimizer: ", optimizer
            print sys.exc_info()[0]

    print "Best parameters: ", best_parameter
    history = {}
    objective(best_parameter)
    results = {}
    
    try:
  
        results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    except:
        results = {}
        results["Exception"] = "Exception"

    print results
    print ""
if PARAMETERS['OPTIMIZER'] == "SLSQP":
    PARAM_BOUND = int(PARAMETERS['PARAM_BOUND'])
    lookup = [x for x in xrange( -int(PARAM_BOUND), int(PARAM_BOUND))]
    
    for idx in xrange(5):
        
        p = [ choice(lookup) for x in xrange(2 + int(PARAMETERS['PARAMETERS']))]
        current_parameters = map( lambda h: 2 ** float(h), p)
        print "Starting from ", current_parameters
        best_parameter = None

        try:
            best_parameter = minimize(min_objective, current_parameters, method="SLSQP")
            best_parameter = best_parameter.x
            print "Success with optimizer: ", optimizer
        except:
            print "Exception with optimizer: ", optimizer
            print sys.exc_info()[0]

    print "Best parameters: ", best_parameter
    history = {}
    objective(best_parameter)
    results = {}
    
    try:

        results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    except:
        results = {}
        results["Exception"] = "Exception"

    print results
    print ""

    
if PARAMETERS['OPTIMIZER'] == "CMA":
    x0 = [1.0] * (2 + int(PARAMETERS['PARAMETERS']))
    res = cma.fmin(min_objective, x0, 25, maxiter='10')
    best_parameter = res[0]
    print "Best parameters: ", best_parameter
    history = {}
    objective(best_parameter)
    results = {}
    try:
        results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    except:
        results = {}
        resuls["Exception"] = "Exception"

    print results
    print ""


if PARAMETERS['OPTIMIZER'] == "CMA":
    x0 = [1.0] * (2 + int(PARAMETERS['PARAMETERS']))
    res = cma.fmin(min_objective, x0, 25, maxiter='10')
    best_parameter = res[0]
    print "Best parameters: ", best_parameter
    history = {}
    objective(best_parameter)
    results = {}
    try:
        results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    except:
        results = {}
        resuls["Exception"] = "Exception"

    print results
    print ""

if PARAMETERS['OPTIMIZER'] == "GaussSouthwell":
    best_parameter, history = gauss_southwell_search(min_objective)
    print "Best parameters: ", best_parameter
    history = {}
    objective(best_parameter)
    results = {}
    try:
        results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    except:
        results = {}
        resuls["Exception"] = "Exception"
    print results
    print ""

# Grid search
if PARAMETERS['OPTIMIZER'] == "GridSearch":
    best_parameter, history, grid = grid_search(objective, PARAMETERS['PARAMETERS'], PARAMETERS['PARAM_BOUND'])
    for k,v in history.iteritems():
        print k, '=>', v

    pickle.dump(grid, open(sys.argv[1] + ".grid", "wb"))
    # print "Best parameters"
    # kernel = classifier.get_kernel()
    # w = 2
    # kernel.get_subkernel_weights(w)
    # print kern.subkernel_weights()
    objective(best_parameter, [x*10 for x in xrange(9)], production=True)
    results = {}
    # try:
    
    #results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    # except:
    #     results = {}
    #     resuls["Exception"] = "Exception"

    # print results
    print ""

    # Grid search
if PARAMETERS['OPTIMIZER'] == "RandomSearch":
    best_parameter, history, grid = random_search(objective, PARAMETERS['PARAMETERS'], PARAMETERS['PARAM_BOUND'])
    for k,v in history.iteritems():
        print k, '=>', v
    pickle.dump(grid, open(sys.argv[1] + ".grid", "wb"))
    print ""
    print "Best parameters"
    
    objective(best_parameter)
    results = {}
    # try:
    results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    # except:
    #     results = {}
    #     resuls["Exception"] = "Exception"

    print results
    print ""

if PARAMETERS['OPTIMIZER'] == "SobolSearch":
    best_parameter, history, grid = sobol_search(objective, PARAMETERS['PARAMETERS'], PARAMETERS['PARAM_BOUND'])
    for k,v in history.iteritems():
        print k, '=>', v
    pickle.dump(grid, open(sys.argv[1] + ".grid", "wb"))
    print ""
    print "Best parameters"
    
    objective(best_parameter)
    results = {}
    # try:
    results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    # except:
    #     results = {}
    #     resuls["Exception"] = "Exception"

    print results
    print ""

if PARAMETERS['OPTIMIZER'] == "LHSSearch":
    best_parameter, history, grid = latin_search(objective, PARAMETERS['PARAMETERS'], PARAMETERS['PARAM_BOUND'])
    for k,v in history.iteritems():
        print k, '=>', v
    pickle.dump(grid, open(sys.argv[1] + ".grid", "wb"))
    print ""
    print "Best parameters"
    
    objective(best_parameter)
    results = {}
    # try:
    results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    # except:
    #     results = {}
    #     resuls["Exception"] = "Exception"

    print results
    print ""


if PARAMETERS['OPTIMIZER'] == "HaltonSearch":
    print "HALTON PARAMETER SERACH"
    best_parameter, history, grid = halton_search(objective, PARAMETERS['PARAMETERS'], PARAMETERS['PARAM_BOUND'])
    for k,v in history.iteritems():
        print k, '=>', v
    pickle.dump(grid, open(sys.argv[1] + ".grid", "wb"))
    print ""
    print "Best parameters"
    
    objective(best_parameter)
    results = {}
    # try:
    results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    # except:
    #     results = {}
    #     resuls["Exception"] = "Exception"

    print results
    print ""

if PARAMETERS['OPTIMIZER'] == "SobolSearch":
    best_parameter, history, grid = sobol_search(objective, PARAMETERS['PARAMETERS'], PARAMETERS['PARAM_BOUND'])
    for k,v in history.iteritems():
        print k, '=>', v
    pickle.dump(grid, open(sys.argv[1] + ".grid", "wb"))
    print ""
    print "Best parameters"
    
    objective(best_parameter)
    results = {}
    # try:
    results = evaluate(best_parameter, kernel, PARAMETERS['KERNEL'], ctype, classifier, train_labels, train_features, int(PARAMETERS['FOLDS']), int(PARAMETERS['RUNS']))
    # except:
    #     results = {}
    #     resuls["Exception"] = "Exception"

    print results
    print ""
# endTime = datetime.datetime.now()
# elapsedTime = endTime - startTime


storage = {}
# storage['elapsedTime'] = elapsedTime
storage['configFile'] = PARAMETERS
storage['history'] = history
storage['bestParameter'] = best_parameter
storage['results'] = results
save(storage, sys.argv[1] + ".result")


print load(sys.argv[1] + ".result")
