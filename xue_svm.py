from numpy import *
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
from pylab import *


import sys
import pickle
import scipy.sparse

#DATA_SOURCE = "data/iri/da.%s"
#DATA_SOURCE = "data/iri/go.%s"
#DATA_SOURCE = "data/iri/ia.%s"
#FEATURES = sys.argv[1]
#LABELS = sys.argv[2]

#DATA_SOURCE = "data/iri/oa.%s"

def evaluation_rocevaluation_modular(ground_truth, predicted):
	from shogun.Features import BinaryLabels
	from shogun.Evaluation import ROCEvaluation

	ground_truth_labels = BinaryLabels(ground_truth)
	predicted_labels = BinaryLabels(predicted)

	evaluator = ROCEvaluation()
	evaluator.evaluate(predicted_labels,ground_truth_labels)

	return evaluator.get_ROC(), evaluator.get_auROC()

def evaluation_contingencytableevaluation_modular(ground_truth, predicted):
	from shogun.Features import BinaryLabels
	from shogun.Evaluation import ContingencyTableEvaluation
	from shogun.Evaluation import AccuracyMeasure,ErrorRateMeasure,BALMeasure
	from shogun.Evaluation import WRACCMeasure,F1Measure,CrossCorrelationMeasure
	from shogun.Evaluation import RecallMeasure,PrecisionMeasure,SpecificityMeasure

	ground_truth_labels = BinaryLabels(ground_truth)
	predicted_labels = BinaryLabels(predicted)
	
	base_evaluator = ContingencyTableEvaluation()
	base_evaluator.evaluate(predicted_labels,ground_truth_labels)

	evaluator = AccuracyMeasure()	
	accuracy = evaluator.evaluate(predicted_labels,ground_truth_labels)

	evaluator = ErrorRateMeasure()
	errorrate = evaluator.evaluate(predicted_labels,ground_truth_labels)

	evaluator = BALMeasure()
	bal = evaluator.evaluate(predicted_labels,ground_truth_labels)

	evaluator = WRACCMeasure()
	wracc = evaluator.evaluate(predicted_labels,ground_truth_labels)

	evaluator = F1Measure()
	f1 = evaluator.evaluate(predicted_labels,ground_truth_labels)

	evaluator = RecallMeasure()
	recall = evaluator.evaluate(predicted_labels,ground_truth_labels)

	evaluator = PrecisionMeasure()
	precision = evaluator.evaluate(predicted_labels,ground_truth_labels)

	evaluator = SpecificityMeasure()
	specificity = evaluator.evaluate(predicted_labels,ground_truth_labels)

	return accuracy, errorrate, bal, wracc, f1, recall, precision, specificity


def decrement(x):
	return x - 1

def clean(str):
	str = str.strip("rowscolsvals: ")
	return str

num=1000
numt = 100
dist=1
C=1
s=2


pkl = open("data/xue/pgp.pkl", "r")
data = pickle.load(pkl)
data, labs = data[0], data[1]

#data = scipy.sparse.coo_matrix( (vals, (rows,cols) ), shape=(n_rows,n_cols))
#print data.shape
#print labs.shape
# Generate some data
#traindata_real = concatenate((randn(2,num)-dist, randn(2,num)+dist),   axis=1)
#print traindata_real
traindata_real = array(data, dtype=float64).T
print traindata_real.shape
#testdata_real  = concatenate((randn(2,numt)-dist, randn(2,numt)+dist), axis=1);

# BinaryLabels
#trainlab = concatenate((-ones(num), ones(num)));
#print trainlab.shape
trainlab = array(labs, dtype=float64)
print trainlab.shape
#testlab  = concatenate((-ones(numt), ones(numt)));

# Split into pos/neg train/test for plotting
trainpos = traindata_real[:,trainlab ==  1]
trainneg = traindata_real[:,trainlab == -1]
#testpos  = testdata_real[:,testlab   ==  1]
#testneg  = testdata_real[:,testlab   == -1]

# Pack labels and features into desnse representation
feats_train = RealFeatures(traindata_real);
#feats_test  = RealFeatures(testdata_real);
labels      = BinaryLabels(trainlab);

# Generate the kernel matrix
kernel=GaussianKernel()
kernel.init(feats_train, feats_train);

# Create a classifier
classifier=LibSVM()
classifier.set_kernel(kernel)
classifier.set_labels(labels)
classifier.set_C(C, C)

classifier.print_modsel_params()
kernel.print_modsel_params()

#### Parameter Estimation
# Generate parameter tree
param_tree_root=ModelSelectionParameters()

# Attached C1 parameter to the tree
c1=ModelSelectionParameters("C1");
c1.build_values(-1.0, 1.0, R_EXP);
param_tree_root.append_child(c1)

# Attached C2 parameter to the tree
c2=ModelSelectionParameters("C2");
c2.build_values(-1, 1.0, R_EXP);
param_tree_root.append_child(c2)

# Add kernel width parameter to the tree
#param_gaussian_kernel=ModelSelectionParameters('kernel', kernel)
#param_gaussian_kernel_width=ModelSelectionParameters('width')
#param_gaussian_kernel_width.build_values(-1.0, 1.0, R_EXP)
#param_gaussian_kernel.append_child(param_gaussian_kernel_width)
#param_tree_root.append_child(param_gaussian_kernel)

splitting_strategy=StratifiedCrossValidationSplitting(labels, 78)
evaluation_criterium=ContingencyTableEvaluation(ACCURACY)
cross_validation=CrossValidation(classifier, feats_train, labels, splitting_strategy, evaluation_criterium)
model_selection=GridSearchModelSelection(param_tree_root, cross_validation)
best_parameters=model_selection.select_model(True)

print "Best parameters: ",
best_parameters.print_tree()
best_parameters.apply_to_machine(classifier)
classifier.train()

print trainlab
print classifier.apply(feats_train).get_labels()

print evaluation_contingencytableevaluation_modular((trainlab), (classifier.apply(feats_train).get_labels()))
#print evaluation_rocevaluation_modular((trainlab), (classifier.apply(feats_train).get_labels()))
# Run the SVM
#subplot(111)
#ROC_evaluation=ROCEvaluation()
#ROC_evaluation.evaluate(classifier.apply(feats_train),BinaryLabels(trainlab))
#roc = ROC_evaluation.get_ROC()
#print roc[0].shape
#print roc[1].shape
#plot(roc[0], roc[1])
#fill_between(roc[0],roc[1],0,alpha=0.1)
#grid(True)
#xlabel('FPR')
#ylabel('TPR')
#title('ROC (Width=%.3f, C1=%.3f, C2=%.3f) ROC curve = %.3f' % (kernel.get_width(), classifier.get_C1(), classifier.get_C2(), ROC_evaluation.get_auROC()),size=10)
#savefig("unamed.png")
"""
subplot(222)
ROC_evaluation=ROCEvaluation()
ROC_evaluation.evaluate(classifier.apply(feats_test),BinaryLabels(testlab))
roc = ROC_evaluation.get_ROC()
plot(roc[0], roc[1])
fill_between(roc[0],roc[1],0,alpha=0.1)
grid(True)
xlabel('FPR')
ylabel('TPR')
title('Test ROC (Width=%.3f, C1=%.3f, C2=%.3f) ROC curve = %.3f' % (kernel.get_width(), classifier.get_C1(), classifier.get_C2(), ROC_evaluation.get_auROC()),size=10)


subplot(223)
plot(trainpos[0, :], trainpos[1, :], "r.")
plot(trainneg[0, :], trainneg[1, :], "b.")
#plot(testpos[0, :], testpos[1, :], "rx")
#plot(testneg[0, :], testneg[1, :], "bx")

grid(True)
title('Training Data',size=10)
x, y, z = compute_output_plot_isolines(classifier, kernel, feats_train)
pcolor(x, y, z, shading='interp')
contour(x, y, z, linewidths=1, colors='black', hold=True)
axis('tight')

subplot(224)
#plot(trainpos[0, :], trainpos[1, :], "r.")
#plot(trainneg[0, :], trainneg[1, :], "b.")
plot(testpos[0, :], testpos[1, :], "r.")
plot(testneg[0, :], testneg[1, :], "b.")

grid(True)
title('Testing Data',size=10)
x, y, z = compute_output_plot_isolines(classifier, kernel, feats_train)
pcolor(x, y, z, shading='interp')
contour(x, y, z, linewidths=1, colors='black', hold=True)
axis('tight')
"""


#show()

