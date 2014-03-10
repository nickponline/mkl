from numpy import *
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
from pylab import *

import sys
import pickle
import scipy.sparse
import numpy

def make_labels(labels):
	return BinaryLabels(labels)

def make_features(data):
	if isinstance(data, scipy.sparse.csc_matrix):
		return SparseRealFeatures(data)
	elif isinstance(data, numpy.ndarray):
		return RealFeatures(data)

def make_kernel(features, type):
	# Single kernel
	if type=="svm":
		kernel=GaussianKernel()
		kernel.init(features[0], features[0])
		return kernel
	# Multiple kernel
	elif type=="mkl":
		kernel = CombinedKernel()
		for i in xrange(5): # 5 kernels
			kernel.append_kernel(LinearKernel())
		kernel.init(features, features)
		return kernel

# Make some fake data
data = numpy.random.uniform(0,1,(6,6))
labs = numpy.array([1, 1, 1, -1, -1, -1], dtype=float64)

# Make some features and labels
#train_features = make_features(data);

train_features = CombinedFeatures()
train_features.append_feature_obj(make_features(data))
train_features.append_feature_obj(make_features(data))
train_features.append_feature_obj(make_features(data))
train_features.append_feature_obj(make_features(data))
train_features.append_feature_obj(make_features(data))

train_labels   = make_labels(labs);

# MKL 5 kernels (doesn't work)
# Gives SystemError: assertion l->get_feature_class()==C_COMBINED failed in file kernel/CombinedKernel.cpp line 70
kernel=make_kernel(train_features, type="mkl")
kernel.print_modsel_params()
classifier=MKLClassification(LibSVM())
classifier.set_interleaved_optimization_enabled(False)
classifier.set_kernel(kernel)
classifier.set_labels(train_labels)
classifier.set_C(1.0, 1.0)
classifier.print_modsel_params()

# Do model selection
param_tree_root=ModelSelectionParameters()
c1=ModelSelectionParameters("C1");
c1.build_values(0.0, 1.0, R_EXP);
param_tree_root.append_child(c1)
c2=ModelSelectionParameters("C2");
c2.build_values(0, 1.0, R_EXP);
param_tree_root.append_child(c2)
splitting_strategy   = StratifiedCrossValidationSplitting(train_labels, 2)
evaluation_criterium = ContingencyTableEvaluation(ACCURACY)
cross_validation     = CrossValidation(classifier, train_features, train_labels, splitting_strategy, evaluation_criterium)
model_selection      = GridSearchModelSelection(param_tree_root, cross_validation)
best_parameters      = model_selection.select_model(True)
best_parameters.apply_to_machine(classifier)

classifier.train()

print classifier.apply(train_features).get_labels()


