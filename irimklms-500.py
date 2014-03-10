from shogun.Features import CombinedFeatures, RealFeatures, Labels
from shogun.Kernel import CombinedKernel, PolyKernel, CustomKernel
from shogun.Classifier import MKLClassification
from numpy import *
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
from pylab import *

import pickle
import scipy.sparse

#go_bp go_cc goip go_mf ip go_bp

DATA_SOURCE_1 = "data/iri/gisIn33_dmel08HcLcT3-500_ip_protId_SVM%s.txt"
DATA_SOURCE_2 = "data/iri/gisIn33_dmel08HcLcT3-500_go_cc_protId_SVM%s.txt"
DATA_SOURCE_3 = "data/iri/gisIn33_dmel08HcLcT3-500_go_mf_protId_SVM%s.txt"
#DATA_SOURCE_4 = "new-data/gisIn33_dmel08HcLcT2-500_go_mf_protId_SVM%s.txt"
#DATA_SOURCE_5 = "new-data/gisIn33_dmel08HcLcT2-500_ip_protId_SVM%s.txt"

def decrement(x):
	return x - 1

def clean(str):
	str = str.strip("rowscolsvals: ")
	return str

def load_iri_data(data_source):
	data = file(data_source % "").read().split("\n")
	data = map(clean, data)
	labs = array(map(lambda x : 1 if x == 1 else -1, map(int, file(data_source % "_labels").read().strip().split("\n"))))
	rows = map(int, data[0].split(","))
	cols = map(int, data[1].split(","))
	vals = array(map(int, data[2].split(",")))
	rows = array(map(decrement, rows))
	cols = array(map(decrement, cols))
	n_rows = len(labs)
	n_cols = cols.max()+1
	data = scipy.sparse.coo_matrix( (vals, (rows,cols) ), shape=(n_rows,n_cols)).todense()
	print data.T.shape
	print labs.shape
	return (array(data.T, dtype=float64), array(labs, dtype=float64))

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

	if kernel and train:
		kernel.init(train, test)
	else:
		classifier.set_features(test)

	labels=classifier.apply().get_labels()
	z=labels.reshape((size, size))
	return x, y, z


# Generate some data
data_1, labs_1 = load_iri_data(DATA_SOURCE_1)
data_2, labs_2 = load_iri_data(DATA_SOURCE_2)
data_3, labs_3 = load_iri_data(DATA_SOURCE_3)

#traindata_real = concatenate((randn(2,num)-dist, randn(2,num)+dist),   axis=1)
#testdata_real  = concatenate((randn(2,numt)-dist, randn(2,numt)+dist), axis=1);

# Labels
trainlab = labs_1
#testlab  = concatenate((-ones(numt), ones(numt)));

# Split into pos/neg train/test for plotting
#trainpos = traindata_real[:,trainlab ==  1]
#trainneg = traindata_real[:,trainlab == -1]
#testpos  = testdata_real[:,testlab   ==  1]
#testneg  = testdata_real[:,testlab   == -1]

# create combined train features
feats_train = CombinedFeatures()
feats_train.append_feature_obj(RealFeatures(data_1))
feats_train.append_feature_obj(RealFeatures(data_2))
feats_train.append_feature_obj(RealFeatures(data_3))

#feats_test = CombinedFeatures()
#feats_test.append_feature_obj(RealFeatures(testdata_real))
#feats_test.append_feature_obj(RealFeatures(testdata_real))
#feats_test.append_feature_obj(RealFeatures(testdata_real))
#feats_test.append_feature_obj(RealFeatures(testdata_real))
#feats_test.append_feature_obj(RealFeatures(testdata_real))

labels = BinaryLabels(trainlab)

# and corresponding combined kernel
kernel = CombinedKernel()
kernel.append_kernel(LinearKernel())
kernel.append_kernel(LinearKernel())
kernel.append_kernel(LinearKernel())
kernel.init(feats_train, feats_train)
kernel.print_modsel_params()

# Create a classifier
classifier=MKLClassification(LibSVM())
classifier.set_interleaved_optimization_enabled(False)
classifier.set_kernel(kernel)
classifier.set_labels(labels)
classifier.set_C(2, 1)

param_tree_root=ModelSelectionParameters()

# () C1 parameter to the tree
c1=ModelSelectionParameters("C1"); 
c1.build_values(-2.0, 2.0, R_EXP);
param_tree_root.append_child(c1)

# Attached C1 parameter to the tree
c2=ModelSelectionParameters("C2");
c2.build_values(-2.0, 2.0, R_EXP);
param_tree_root.append_child(c2)

splitting_strategy   = StratifiedCrossValidationSplitting(labels, 5)
evaluation_criterium = ContingencyTableEvaluation(ACCURACY)
cross_validation     = CrossValidation(classifier, feats_train, labels, splitting_strategy, evaluation_criterium)
model_selection      = GridSearchModelSelection(param_tree_root, cross_validation)

print "STARTING"
best_parameters      = model_selection.select_model(True)

print "Best parameters: ",
#best_parameters.print_tree()
#best_parameters.apply_to_machine(classifier)

classifier.train()
w=kernel.get_subkernel_weights()
kernel.set_subkernel_weights(w)

# Plot ROC curve
subplot(111)
ROC_evaluation=ROCEvaluation()
ROC_evaluation.evaluate(classifier.apply(feats_train),Labels(trainlab))
roc = ROC_evaluation.get_ROC()
plot(roc[0], roc[1])
fill_between(roc[0],roc[1],0,alpha=0.1)
grid(True)
xlabel('FPR')
ylabel('TPR')
title('Train ROC (Width=%.3f, C1=%.3f, C2=%.3f) ROC curve = %.3f' % (10, classifier.get_C1(), classifier.get_C2(), ROC_evaluation.get_auROC()),size=10)
savefig("data/iri/mkl.png")
"""
subplot(222)
ROC_evaluation=ROCEvaluation()
ROC_evaluation.evaluate(classifier.apply(feats_test),Labels(testlab))
roc = ROC_evaluation.get_ROC()
plot(roc[0], roc[1])
fill_between(roc[0],roc[1],0,alpha=0.1)
grid(True)
xlabel('FPR')
ylabel('TPR')
title('Test ROC (Width=%.3f, C1=%.3f, C2=%.3f) ROC curve = %.3f' % (0, classifier.get_C1(), classifier.get_C2(), ROC_evaluation.get_auROC()),size=10)



subplot(223)
plot(trainpos[0, :], trainpos[1, :], "g.")
plot(trainneg[0, :], trainneg[1, :], "r.")
grid(True)
title('Training Data',size=10)
x, y, z = compute_output_plot_isolines(classifier, kernel, feats_train)
pcolor(x, y, z, shading='interp')
contour(x, y, z, linewidths=1, colors='black', hold=True)
axis('tight')

subplot(224)
plot(testpos[0, :], testpos[1, :], "g.")
plot(testneg[0, :], testneg[1, :], "r.")
grid(True)
title('Testing Data',size=10)
x, y, z = compute_output_plot_isolines(classifier, kernel, feats_train)
pcolor(x, y, z, shading='interp')
contour(x, y, z, linewidths=1, colors='black', hold=True)
axis('tight')
"""

#show()


   
