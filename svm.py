from numpy import *
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
from pylab import *

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

num=100
dist=1
C=1
s=2

# Generate some data
seed(42)
traindata_real = concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1)
testdata_real  = concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1);

print traindata_real
# Labels
trainlab = concatenate((-ones(num), ones(num)));
testlab  = concatenate((-ones(num), ones(num)));

# Split into pos/neg train/test for plotting
trainpos = traindata_real[:,trainlab ==  1]
trainneg = traindata_real[:,trainlab == -1]
testpos  = testdata_real[:,testlab   ==  1]
testneg  = testdata_real[:,testlab   == -1]

# Pack labels and features into desnse representation
feats_train = RealFeatures(traindata_real);
feats_test  = RealFeatures(testdata_real);
labels      = Labels(trainlab);

# Generate the kernel matrix
kernel=GaussianKernel(10, s)
kernel.init(feats_train, feats_test);
kernel.print_modsel_params()

#### Parameter Estimation
# Generate parameter tree
param_tree_root=ModelSelectionParameters()

# Attached C1 parameter to the tree
c1=ModelSelectionParameters("C1");
c1.build_values(-2, 2.0, R_EXP);
param_tree_root.append_child(c1)

# Attached C2 parameter to the tree
c2=ModelSelectionParameters("C2");
c2.build_values(-2, 2.0, R_EXP);
param_tree_root.append_child(c2)

# Add kernel width parameter to the tree
param_gaussian_kernel=ModelSelectionParameters('kernel', kernel)
param_gaussian_kernel_width=ModelSelectionParameters('width')
param_gaussian_kernel_width.build_values(-2, 2.0, R_EXP)
param_gaussian_kernel.append_child(param_gaussian_kernel_width)
param_tree_root.append_child(param_gaussian_kernel)

# Create a classifier
classifier=LibSVM()
classifier.set_kernel(kernel)
classifier.set_labels(labels)

# 10-fold stratified cross validation
splitting_strategy=StratifiedCrossValidationSplitting(labels, 50)
# evaluation method
evaluation_criterium=ContingencyTableEvaluation(ACCURACY)
# cross-validation instance
cross_validation=CrossValidation(classifier, feats_train, labels, splitting_strategy, evaluation_criterium)
# model selection instance
model_selection=GridSearchModelSelection(param_tree_root, cross_validation)
# perform model selection with selected methods
best_parameters=model_selection.select_model(True)

print "Best parameters: ",
best_parameters.print_tree()

#apply them and print result
best_parameters.apply_to_machine(classifier)

# Run the SVM
subplot(121)
ROC_evaluation=ROCEvaluation()
ROC_evaluation.evaluate(classifier.apply(),labels)
roc = ROC_evaluation.get_ROC()
plot(roc[0], roc[1])
fill_between(roc[0],roc[1],0,alpha=0.1)
text(mean(roc[0])/2,mean(roc[1])/2,'auROC = %.5f' % ROC_evaluation.get_auROC())
print "ROC: ", ROC_evaluation.get_auROC()
grid(True)
xlabel('FPR')
ylabel('TPR')
title('SVM (Gaussian kernel, C1=%.3f, C2=%.3f) ROC curve' % (classifier.get_C1(), classifier.get_C2()),size=10)

subplot(122)
plot(trainpos[0, :], trainpos[1, :], "r.")
plot(trainneg[0, :], trainneg[1, :], "b.")
plot(testpos[0, :], testpos[1, :], "rx")
plot(testneg[0, :], testneg[1, :], "bx")

grid(True)
title('Data',size=10)
x, y, z = compute_output_plot_isolines(classifier, kernel, feats_train)
pcolor(x, y, z, shading='interp')
contour(x, y, z, linewidths=1, colors='black', hold=True)
axis('tight')



show()

