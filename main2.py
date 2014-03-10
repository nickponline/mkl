import cma
import csv
import shelve
import datetime
import random
import sys
import pickle
import numpy

from numpy import *
from numpy.random import randn
from numpy.testing import *

from scipy.optimize import *
from scipy.sparse import *

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


def set_inner_kernel_parameters(kernel, w):

    # Only current consider Gaussian and linear kernels
    if kernel.get_name() == "GaussianKernel":
        kernel.set_width(w)
    elif kernel.get_name() == "WeightedDegreeRBFKernel":
        kernel.set_width(w)
    else:
        pass


def set_kernel_parameters(kernel, w):

    if kernel.get_name() == "CombinedKernel":
        for i in xrange(kernel.get_num_subkernels()): 
            if kernel.get_kernel(i).get_name() == "GaussianKernel":
                gaussian_kernel = GaussianKernel.obtain_from_generic(kernel.get_kernel(i))
                gaussian_kernel.set_width(w)
            else:
                pass
    else:
        set_inner_kernel_parameters(kernel, w)


def objective(parameters):

    Math_init_random(17)

    c1, c2, w = parameters

    set_kernel_parameters(kernel, w)
    
    classifier.set_C(c1, c2)
    
    splitting_strategy   = StratifiedCrossValidationSplitting(train_labels, FOLDS)

    # splitting_strategy.build_subsets()
    # subs = splitting_strategy.get_num_subsets()
    # for i in xrange(subs):
    #     train_idx = splitting_strategy.generate_subset_indices(i)
    #     test_idx = splitting_strategy.generate_subset_inverse(i)
    #     train_data = train

    evaluation_criterium = ContingencyTableEvaluation(F1)
    cross_validation     = CrossValidation(classifier, train_features, train_labels, splitting_strategy, evaluation_criterium)
        
    cross_validation.set_num_runs(RUNS) 
    cross_validation.set_conf_int_alpha(0.05)
    
    #cross_validation.set_autolock(False)
    
    result = cross_validation.evaluate()
    
    fmt_parameters = map(lambda h: "%3.3f" % h, parameters)
    print "Obj => %2.5f" % result.get_mean(), "Args => ",fmt_parameters

    return result.get_mean()

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




PARAM_BOUND, FOLDS, RUNS, DATASET, METHOD, FEATURES       = load_configuration(sys.argv[1])
data, labs, ctype, K                                      = load_data(DATASET)

data, K, ctype                                            = convolve_features(data, K, ctype)
data                                                      = map(normalizer, data)

print "Kernels: ", K
# data, labs, ctype, K                                      = make_data(5, 200, 500, 2)
# data                                                       = make_feature_select(data, labs, FEATURES)
# training_data, testing_data, training_labs, testing_labs = make_training_test(data, labs)

train_features                                            = make_features(data);
train_labels                                              = make_labels(labs);

# test_features                                            = make_features(testing_data);
# test_labels                                              = make_labels(testing_labs);

kernel                                                    = make_kernel(K, train_features, ctype=ctype, widths=[1.0] * K)
classifier                                                = make_classifier(train_features, train_labels, kernel, ctype=ctype)


# for k in xrange(, 6):
#     print classifier.get_kernel(k)



#### Parameter Estimation + Training
# Generate parameter tree
#param_tree_root=ModelSelectionParameters()
# Attached C1 parameter to the tree
#c1=ModelSelectionParameters("C1");
#c1.build_values(-PARAM_BOUND, PARAM_BOUND, R_EXP);
#param_tree_root.append_child(c1)
# Attached C2 parameter to the tree
##c2=ModelSelectionParameters("C2");
#c2.build_values(-PARAM_BOUND, PARAM_BOUND, R_EXP);
#param_tree_root.append_child(c2)
# Attached C3 parameter to the tree
#c3=ModelSelectionParameters("nu");
#c3.build_values(-3.0, 3.0, R_EXP);
#param_tree_root.append_child(c3)

# print "Optimizing with bfgs"
# x0 = [500, 500]
# res = fmin_l_bfgs_b(objective, x0, approx_grad=True, bounds=[(1.0/1000.0, 1000), (1.0/1000.0, 1000)], maxfun=10, pgtol=1e-3)
# print "Best parameters: ", res[0]
# C1, C2 = map(abs, res[0])
# classifier.set_C(C1, C2)
# ret = evaluate(ctype, classifier, train_labels, train_features, FOLDS, RUNS)
# print ""

# print "Optimizing with tnc"
# x0 = [500, 500]
# res = fmin_tnc(objective, x0, approx_grad=True, bounds=[(1.0/1000.0, 1000), (1.0/1000.0, 1000)], maxfun=10, pgtol=1e-3)
# print "Best parameters: ", res[0]
# C1, C2 = map(abs, res[0])
# classifier.set_C(C1, C2)
# ret = evaluate(ctype, classifier, train_labels, train_features, FOLDS, RUNS)
# print ""

# print "Optimizing with cobyla"
# x0 = [500, 500]
# res = fmin_cobyla(objective, x0, cons=[lambda x: x[0],lambda x: x[1],lambda x: x[0] - 1000,lambda x: x[1] - 1000, ], maxfun=10, rhobeg=100.0)
# print "Best parameters: ", res
# C1, C2 = map(abs, res)
# classifier.set_C(C1, C2)
# ret = evaluate(ctype, classifier, train_labels, train_features, FOLDS, RUNS)
# print ""

# print "Optimizing with slsqp"
# x0 = [500, 500]
# res = fmin_slsqp(objective, x0, bounds=[(0, 1000), (0, 1000)], acc=1e-3, iter=10)
# print "Best parameters: ", res
# C1, C2 = map(abs, res)
# classifier.set_C(C1, C2)
# ret = evaluate(ctype, classifier, train_labels, train_features, FOLDS, RUNS)
# print ""

#print "Optimizing with anneal"
#x0 = [500, 500]
#res = anneal(objective, x0, maxiter=10, lower=[0, 0], upper=[1000,1000])
#print "Best parameters: ", res
#C1, C2 = map(abs, res)
#classifier.set_C(C1, C2)
#ret = evaluate(ctype, classifier, train_labels, train_features, FOLDS, RUNS)
#print ""

#print "Optimizing with cma"
#x0 = [500, 600]
#res = cma.fmin(objective, x0, 250, maxiter='100')
#print "Best parameters: ", res[0]
#C1, C2 = map(abs, res[0])
#classifier.set_C(C1, C2)
#ret = evaluate(ctype, classifier, train_labels, train_features, FOLDS, RUNS)
#print ""

#x0 = [500, 600]
#res = gauss_stockwell(objective, x0, 250)
#print res

# Grid search
# print "Optimizing with Gaussian Process Surrogate"
# res = gpo1d(objective)
# print "Best parameters: ", res
# C1, C2 = map(abs, res)
# classifier.set_C(C1, C2)
# ret = evaluate(ctype, classifier, train_labels, train_features, FOLDS, RUNS)
# print ""

# classifier.train()
# svs = classifier.get_num_support_vectors()

# for i in xrange(svs):
#     print classifier.get_alpha(i), classifier.get_support_vector(i)

# print "Bias: ", classifier.get_bias()


# Grid search
best_parameter, history = grid_search(objective)
for k,v in history.iteritems():
    print k, '=>', v

print ""
print "Best parameters"
objective(best_parameter)
# evaluate(best_parameter, kernel, ctype, classifier, test_labels, test_features, FOLDS, RUNS)
print ""

