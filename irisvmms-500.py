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
FEATURES = sys.argv[1]
LABELS = sys.argv[2]

#DATA_SOURCE = "data/iri/oa.%s"

def decrement(x):
	return x - 1

def clean(str):
	str = str.strip("rowscolsvals: ")
	return str

def load_iri_data():
	data = file(FEATURES).read().split("\n")
	data = map(clean, data)
	labs = array(map(lambda x : 1 if x == 1 else -1, map(int, file(LABELS).read().strip().split("\n"))))
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
	return (data.T, labs)

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
		test=dense

	if kernel and train:
		kernel.init(train, test)
	else:
		classifier.set_features(test)

	labels=classifier.apply().get_labels()
	z=labels.reshape((size, size))

	#print x.shape, y.shape, z.shape

	return x, y, z

num=1000
numt = 100
dist=1
C=1
s=2

data, labs = load_iri_data()

# Generate some data
#traindata_real = concatenate((randn(2,num)-dist, randn(2,num)+dist),   axis=1)
#print traindata_real
traindata_real = array(data, dtype=float64)
#print traindata_real
#testdata_real  = concatenate((randn(2,numt)-dist, randn(2,numt)+dist), axis=1);

# Labels
#trainlab = concatenate((-ones(num), ones(num)));
#print trainlab.shape
trainlab = array(labs, dtype=float64)
#print trainlab.shape
#testlab  = concatenate((-ones(numt), ones(numt)));

# Split into pos/neg train/test for plotting
trainpos = traindata_real[:,trainlab ==  1]
trainneg = traindata_real[:,trainlab == -1]
#testpos  = testdata_real[:,testlab   ==  1]
#testneg  = testdata_real[:,testlab   == -1]

# Pack labels and features into desnse representation
feats_train = RealFeatures(traindata_real);
#feats_test  = RealFeatures(testdata_real);
labels      = Labels(trainlab);

# Generate the kernel matrix
kernel=LinearKernel()
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
c1.build_values(-2.0, 2.0, R_EXP);
param_tree_root.append_child(c1)

# Attached C2 parameter to the tree
c2=ModelSelectionParameters("C2");
c2.build_values(-2, 2.0, R_EXP);
param_tree_root.append_child(c2)

# Add kernel width parameter to the tree
#param_gaussian_kernel=ModelSelectionParameters('kernel', kernel)
#param_gaussian_kernel_width=ModelSelectionParameters('width')
#param_gaussian_kernel_width.build_values(-2.0, 2.0, R_EXP)
#param_gaussian_kernel.append_child(param_gaussian_kernel_width)
#param_tree_root.append_child(param_gaussian_kernel)

splitting_strategy=StratifiedCrossValidationSplitting(labels, 25)
evaluation_criterium=ContingencyTableEvaluation(ACCURACY)
cross_validation=CrossValidation(classifier, feats_train, labels, splitting_strategy, evaluation_criterium)
model_selection=GridSearchModelSelection(param_tree_root, cross_validation)
best_parameters=model_selection.select_model(True)

print "Best parameters: ",
best_parameters.print_tree()
best_parameters.apply_to_machine(classifier)
classifier.train()

# Run the SVM
subplot(111)
ROC_evaluation=ROCEvaluation()
ROC_evaluation.evaluate(classifier.apply(feats_train),Labels(trainlab))
roc = ROC_evaluation.get_ROC()
plot(roc[0], roc[1])
fill_between(roc[0],roc[1],0,alpha=0.1)
grid(True)
xlabel('FPR')
ylabel('TPR')
title('ROC (Width=%.3f, C1=%.3f, C2=%.3f) ROC curve = %.3f' % (0, classifier.get_C1(), classifier.get_C2(), ROC_evaluation.get_auROC()),size=10)
savefig(FEATURES + ".png")
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

