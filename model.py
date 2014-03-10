# In this example simple crossvalidation model parameters selection is shown.
# During the first step a model parameter tree with values ranges of 
# C1 and C2 parameters of SVM is constructed. After that a splitting
# strategy, an evaluation criterium and model selection instance
# are set

#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2011 Heiko Strathmann
# Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
#

from numpy.random import randn
from numpy import *
from shogun.Evaluation import CrossValidation, CrossValidationResult
from shogun.Evaluation import ContingencyTableEvaluation, ACCURACY
from shogun.Evaluation import StratifiedCrossValidationSplitting
from shogun.ModelSelection import GridSearchModelSelection
from shogun.ModelSelection import ModelSelectionParameters, R_EXP
from shogun.ModelSelection import ParameterCombination
from shogun.Features import Labels
from shogun.Features import RealFeatures
from shogun.Classifier import *
from shogun.Kernel import GaussianKernel

# generate some overlapping training vectors
num_vectors=100
vec_distance=1
traindat=concatenate((randn(2,num_vectors)-vec_distance, randn(2,num_vectors)+vec_distance), axis=1)
label_traindat=concatenate((-ones(num_vectors), ones(num_vectors)));
	
# build parameter tree to select C1 and C2 


#c2=ModelSelectionParameters("C2");
#param_tree_root.append_child(c2);
#c2.build_values(-2.0, 2.0, R_EXP);

#nu=ModelSelectionParameters("nu");
#param_tree_root.append_child(nu);
#nu.build_values(-2.0, 2.0, R_EXP);

# training data
features=RealFeatures(traindat)
labels=Labels(label_traindat)

# kernel
# width for the kernel
width  = 2.0
kernel = GaussianKernel(features, features, width)

# Generate parameter tree
param_tree_root=ModelSelectionParameters()

# Add SVM margin
c1=ModelSelectionParameters("C1");
c1.build_values(-2.0, 2.0, R_EXP);
param_tree_root.append_child(c1)

c2=ModelSelectionParameters("C2");
c2.build_values(-2.0, 2.0, R_EXP);
param_tree_root.append_child(c2)

# Add kernel width parameter
param_gaussian_kernel=ModelSelectionParameters('kernel', kernel)
param_gaussian_kernel_width=ModelSelectionParameters('width')
param_gaussian_kernel_width.build_values(-2.0, 2.0, R_EXP)
param_gaussian_kernel.append_child(param_gaussian_kernel_width)
param_tree_root.append_child(param_gaussian_kernel)

# classifier
classifier=LibSVM()
classifier.set_kernel(kernel)
classifier.set_labels(labels)

# splitting strategy for cross-validation
splitting_strategy=StratifiedCrossValidationSplitting(labels, 10)
# evaluation method
evaluation_criterium=ContingencyTableEvaluation(ACCURACY)
# cross-validation instance
cross_validation=CrossValidation(classifier, features, labels, splitting_strategy, evaluation_criterium)
# model selection instance
model_selection=GridSearchModelSelection(param_tree_root, cross_validation)
# perform model selection with selected methods
print "performing model selection of"
param_tree_root.print_tree()
print "before select model"
best_parameters=model_selection.select_model(True)
print "after select model"
# print best parameters
print "best parameters:"
best_parameters.print_tree()

# apply them and print result
best_parameters.apply_to_machine(classifier)

results = cross_validation.evaluate()
results.print_result()
print results.conf_int_low
print results.conf_int_up
print results.conf_int_alpha




