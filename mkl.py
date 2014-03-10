from shogun.Features import CombinedFeatures, RealFeatures, BinaryLabels
from shogun.Kernel import CombinedKernel, PolyKernel, CustomKernel
from shogun.Classifier import MKLClassification
from numpy import *
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
from pylab import *
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

def compute_output_plot_isolines(classifier, kernel=None, train=None, sparse=False, pos=None, neg=None):
	size=100
	if pos is not None and neg is not None:
		x1_max=max(1.2*pos[0,:])
		x1_min=min(1.2*neg[0,:])
		x2_min=min(1.2*neg[1,:])
		x2_max=max(1.2*pos[1,:])
		x1=linspace(x1_min, x1_max, size)
		x2=linspace(x2_min, x2_max, size)
	else:
		x1=linspace(-5, 5, size)
		x2=linspace(-5, 5, size)

	x, y=meshgrid(x1, x2)

	dense=RealFeatures(array((ravel(x), ravel(y))))
	if sparse:
		test=SparseRealFeatures()
		test.obtain_from_simple(dense)
	else:
		#test=dense
		test = CombinedFeatures()
		test.append_feature_obj(dense)
		test.append_feature_obj(dense)
		test.append_feature_obj(dense)
		test.append_feature_obj(dense)
		test.append_feature_obj(dense)

	if kernel and train:
		kernel.init(train, test)
	else:
		classifier.set_features(test)

	labels=classifier.apply().get_labels()
	z=labels.reshape((size, size))
	return x, y, z


num=1000
dist=1
C=1
s=2

# Generate some data
traindata_real=concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1)
testdata_real=concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1);

# BinaryLabels
trainlab=concatenate((-ones(num), ones(num)));
testlab=concatenate((-ones(num), ones(num)));

pos = traindata_real[:,trainlab ==  1]
neg = traindata_real[:,trainlab == -1]

# create combined train features
feats_train = CombinedFeatures()
feats_train.append_feature_obj(RealFeatures(traindata_real))
feats_train.append_feature_obj(RealFeatures(traindata_real))
feats_train.append_feature_obj(RealFeatures(traindata_real))
feats_train.append_feature_obj(RealFeatures(traindata_real))
feats_train.append_feature_obj(RealFeatures(traindata_real))

# and corresponding combined kernel
kernel = CombinedKernel()
kernel.append_kernel(GaussianKernel(10, s))
kernel.append_kernel(GaussianKernel(10, s))
kernel.append_kernel(GaussianKernel(10, s))
kernel.append_kernel(GaussianKernel(10, s))
kernel.append_kernel(GaussianKernel(10, s))
kernel.init(feats_train, feats_train)
kernel.print_modsel_params()

# train mkl
labels = BinaryLabels(trainlab)
mkl = MKLClassification()

# which norm to use for MKL
mkl.set_mkl_norm(1) #2,3
# set cost (neg, pos)
mkl.set_C(C, C)

# set kernel and labels
mkl.set_kernel(kernel)
mkl.set_labels(labels)

# train
mkl.train()
w=kernel.get_subkernel_weights()
kernel.set_subkernel_weights(w)

# create combined test features
feats_test = CombinedFeatures()
feats_test.append_feature_obj(RealFeatures(testdata_real))
feats_test.append_feature_obj(RealFeatures(testdata_real))
feats_test.append_feature_obj(RealFeatures(testdata_real))
feats_test.append_feature_obj(RealFeatures(testdata_real))
feats_test.append_feature_obj(RealFeatures(testdata_real))

# and corresponding combined kernel
kernel = CombinedKernel()
kernel.append_kernel(GaussianKernel(10, s))
kernel.append_kernel(GaussianKernel(10, s))
kernel.append_kernel(GaussianKernel(10, s))
kernel.append_kernel(GaussianKernel(10, s))
kernel.append_kernel(GaussianKernel(10, s))
kernel.init(feats_train, feats_test)

# and classify
mkl.set_kernel(kernel)
out = mkl.apply().get_labels()

print w
print sign(out)
print testlab

# Plot ROC curve
figure()
ROC_evaluation=ROCEvaluation()
ROC_evaluation.evaluate(mkl.apply(),labels)
roc = ROC_evaluation.get_ROC()
print roc
plot(roc[0], roc[1])
fill_between(roc[0],roc[1],0,alpha=0.1)
text(mean(roc[0])/2,mean(roc[1])/2,'auROC = %.5f' % ROC_evaluation.get_auROC())
grid(True)
xlabel('FPR')
ylabel('TPR')
title('MKL PolyNomial Kernel (C=%.3f) ROC curve' % mkl.get_C1(),size=10)

figure()
plot(pos[0, :], pos[1, :], "g.")
plot(neg[0, :], neg[1, :], "r.")
grid(True)
title('Data',size=10)
x, y, z = compute_output_plot_isolines(mkl, kernel, feats_train)
pcolor(x, y, z, shading='interp')
contour(x, y, z, linewidths=1, colors='black', hold=True)
axis('tight')
show()

show()

    