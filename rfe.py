import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV, RFE
from sklearn.datasets import make_classification
from sklearn.metrics import zero_one
from scipy import sparse

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,
                           n_features=5000,
                           n_informative=3,
                           n_redundant=2,
                           n_repeated=0,
                           n_classes=8,
                           n_clusters_per_class=1,
                           random_state=0)

X_sparse = sparse.csr_matrix(X)
print X.shape, "x", y.shape

# sparse model
clf_sparse = SVC(kernel="linear")
rfe_sparse = RFE(estimator=clf_sparse, n_features_to_select=4, step=0.20)
rfe_sparse.fit(X_sparse, y)
X_r_sparse = rfe_sparse.transform(X_sparse)

print X_r_sparse.shape