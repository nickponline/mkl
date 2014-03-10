from numpy import *
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
from pylab import *


import sys
import pickle
import scipy.sparse

rows = [0, 1, 2]
cols = [0, 1, 2]
vals = [1, 1, 1]
n_rows = 3
n_cols = 3

labs = array([1, 1, -1], dtype=float64)
data = scipy.sparse.coo_matrix( (vals, (rows,cols) ), shape=(n_rows,n_cols), dtype=float64)

traindata_real = scipy.sparse.csc_matrix(data.T)

trainlab = labs

feats_train = SparseRealFeatures(traindata_real);
labels      = BinaryLabels(trainlab);

kernel=GaussianKernel()
kernel.init(feats_train, feats_train);

classifier=LibSVM()
classifier.set_kernel(kernel)
classifier.set_labels(labels)
classifier.set_C(1.0, 1.0)
classifier.train()
