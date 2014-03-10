import numpy as np
import pylab as pl

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif

###############################################################################
# import some data to play with

# The IRIS dataset
iris = datasets.load_iris()

# Some noisy data not correlated
E = np.random.normal(size=(len(iris.data), 35))

# Add the noisy data to the informative features
x = np.array([[0.1, 0, 0, 1, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 0], [1, 3, 4, 1, 2, 2, 4, 5, 6, 2, 1, 2, 5, 6, 7, 3, 6]]).T
y = np.array([0, 0, 0, 1, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 0])

print x
print y

###############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(x, y)
scores = -np.log10(selector.scores_)
scores /= scores.max()
print scores