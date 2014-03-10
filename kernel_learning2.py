import sys
import numpy
import random
import pickle
import pickle
import itertools
import scipy.sparse
import scipy.ndimage

from numpy import *
from numpy.random import randn

from shogun.Classifier import *
from shogun.Mathematics import *
from shogun.Kernel import *
from shogun.Features import *
from shogun.Evaluation import *

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.gaussian_process import GaussianProcess
from sklearn.datasets import make_classification

from pylab import *
from collections import namedtuple

def evaluation_contingencytable_evaluation_modular(ground_truth, predicted):
	
	ground_truth_labels = (ground_truth)
	predicted_labels = (predicted)
	
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
	V  =  "evaluation\naccuracy:\t%f\nerror rate:\t%f\nbal:\t\t%f\nwracc:\t\t%f\nf1:\t\t%f\nrecall:\t\t%f\nprecision:\t%f\nspecificty:\t%f\n"
	return V % (accuracy, errorrate, bal, wracc, f1, recall, precision, specificity)

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

	dense=SparseRealFeatures(array((ravel(x), ravel(y))))
	if sparse:
		test=SparseSparseRealFeatures()
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

def load_data(paths):
	print "Loading data ... "
	ret = []
	rey = None
	ctype = "SKL" if len(paths) == 1 else "MKL"
	K = len(paths)
	for path in paths:
		pkl = open(path, "r")
		X, y, g, notes = pickle.load(pkl)
		#print "data ...", path

		# print notes

		r, c = X.nonzero()
		a, b = X.shape
		print " X =", X.shape, "\ty =", y.shape, "\tSparsity =", 1.0 * r.shape[0] / (a*b)
		#rint "Notes: ", notes
		#rint "X:", X.shape
		#rint "y:", y.shape
		#return X, y
		ret.append(X)
		rey = y
	return ret, rey, ctype, K

def make_labels(labels):
	return BinaryLabels(labels)


def make_training_test(data, labs):

	training_data, testing_data = [], []
	training_labs, testing_labs = [], []
	print ""
	for i, d in enumerate(data):

		rows, cols = d.shape
		ratio = np.floor(cols * 0.8)
		perm = np.random.permutation(cols)

		training_idx = perm[0:ratio]
		testing_idx = perm[ratio:]

		training_data.append(d[:, training_idx])
		testing_data.append(d[:, testing_idx])

		training_labs = labs[training_idx]
		testing_labs = labs[testing_idx]

		print " Kernel = %d" % i

		r, c = training_data[-1].nonzero()
		a, b = training_data[-1].shape
		print " Xtr =", training_data[-1].shape, "\tytr =", training_labs.shape, "\tSparsity =", 1.0 * r.shape[0] / (a*b)

		r, c = testing_data[-1].nonzero()
		a, b = testing_data[-1].shape
		print " Xte =", testing_data[-1].shape, "\tyte =", testing_labs.shape, "\tSparsity =", 1.0 * r.shape[0] / (a*b)

	return training_data, testing_data, training_labs, testing_labs



def make_features(data):
	if len(data) == 1:
		if isinstance(data[0], scipy.sparse.csc_matrix):
			#print " Data is csc sparse."
			return SparseRealFeatures(data[0])
		elif isinstance(data[0], scipy.sparse.coo_matrix):
			#print " Data is coo sparse."
			return SparseRealFeatures(data[0].tocsc())
		elif isinstance(data[0], numpy.ndarray):
			#print " Data is dense."
			return RealFeatures(data[0])
	else:
		features = CombinedFeatures()
		for dt in data:
			sub = None
			if isinstance(dt, scipy.sparse.csc_matrix):
				sub = SparseRealFeatures(dt)
			elif isinstance(dt, numpy.ndarray):
				sub = RealFeatures(dt)
			features.append_feature_obj(sub)
		return features;

def make_kernel(K, features, ctype, widths):
	# Single kernel
	#print "Here"
	assert(K == len(widths))

	if ctype=="SKL":
		#kernel=LinearKernel()
		kernel=GaussianKernel()
		#kernel=GaussianARDKernel()
		
		#kernel.set_normalizer(TanimotoKernelNormalizer(1.2))
		kernel.init(features, features)
		#kernel.print_modsel_params()
		return kernel
	# Multiple kernel
	elif ctype=="MKL":
		kernel = CombinedKernel()
		for i in xrange(K): # 5 kernels
			#subkernel = LinearKernel()
			subkernel = GaussianKernel()
			#subkernel=GaussianARDKernel()
			
			# subkernel.set_normalizer(TanimotoKernelNormalizer())
			kernel.append_kernel(subkernel)
		
		kernel.init(features, features)
		#kernel.print_modsel_params()
		return kernel

def make_classifier(features, labels, kernel, ctype="MKL"):
	print ""
	print "Classifier type ...", ctype.lower()
	print ""
	if(ctype == "MKL"):
		classifier=MKLClassification(LibSVM())
		classifier.set_interleaved_optimization_enabled(False)
		classifier.set_linadd_enabled(False)
		classifier.set_kernel(kernel)
		classifier.set_labels(labels)
		classifier.set_mkl_norm(2.0)
		classifier.set_C(1,1)
		#classifier.print_modsel_params()
		return classifier
	elif(ctype=="SKL"):
		classifier=LibSVM()
		classifier.set_kernel(kernel)
		classifier.set_labels(labels)
		classifier.set_C(1,1)
		#classifier.print_modsel_params()
		return classifier

		
def mkchi2(k):
	"""Make k-best chi2 selector"""
	return 

def make_feature_select(data, y, k):
	
	print "feature-reduction ...", k
	if k == 1:
		return data
	
	for i in xrange(len(data)):
		
		X = data[i].T
		r, c = data[i].shape
		X = SelectKBest(chi2, k=max(1, int(1.0 * r * k))).fit_transform(X, y)
		X = X.T
		data[i] = X
		
	return data

def fix_result(result):
	#[0.889082,0.901585] with alpha=0.050000, mean=0.895333
	bounds = result.split(" ")[0]
	alpha = result.split(" ")[1]
	mean = result.split(" ")[2]
	bounds = bound.strip("[]").split(",")
	alpha = alpha.strip(",").split("=")[0]
	mean = mean.splot("=")[1]
	return bounds, alpha, mean

def evaluate(bestVector, kernel, ctype, classifier, train_labels, train_features, folds, runs, alpha = 0.05):


	c1, c2, sigma = map(abs, bestVector)
	classifier.set_C(c1, c2)

	for i in xrange(kernel.get_num_subkernels()): 
		gaussian_kernel = GaussianKernel.obtain_from_generic(kernel.get_kernel(i))
		gaussian_kernel.set_width(sigma)

	headers = []
	ret = []
	# for code, name in [(0, "ACCURACY"), (10, "ERROR_RATE"), (20, "BAL"), (30, "WRACC"), (40, "F1"), (50, "CROSS_CORRELATION"), (60, "RECALL"), (70, "PRECISION"), (80, "SPECIFICITY")]:
	for code, name in [(40, "F1")]:
		splitting_strategy   = StratifiedCrossValidationSplitting(train_labels, folds)
		evaluation_criterium = ContingencyTableEvaluation(code)
		cross_validation	 = CrossValidation(classifier, train_features, train_labels, splitting_strategy, evaluation_criterium)
		
		cross_validation.set_num_runs(runs) 
		cross_validation.set_conf_int_alpha(alpha)
		cross_validation.set_autolock(False)

		# append cross vlaidation output classes
		weights = []
		if(ctype == "MKL"):
			#cross_validation.add_cross_validation_output(CrossValidationPrintOutput())
			mkl_storage=CrossValidationMKLStorage()
			cross_validation.add_cross_validation_output(mkl_storage)
	
			# perform cross-validation
			result=cross_validation.evaluate()

			# print mkl weights
			weights=mkl_storage.get_mkl_weights()
			#print "mkl weights during cross--validation"
			#print weights

			std = result.get_conf_int_up() - result.get_mean()
			print  name.lower(), "\t\t= %3.3f" % result.get_mean(), "+/- %3.3f p=%3.3f" % (std, alpha)
			print "mean-weights     =" , map(lambda h: "%3.3f" % h, Statistics.matrix_mean(weights, False))
			# print "variance-weights =" , map(lambda h: "%3.3f" % h, Statistics.matrix_variance(weights, False))
			print "std-weights      =" , map(lambda h: "%3.3f" % h, Statistics.matrix_std_deviation(weights, False))

		else:
			#cross_val_output = CrossValidationPrintOutput()
			#cross_validation.add_cross_validation_output(cross_val_output)
			result  = cross_validation.evaluate()
			std = result.get_conf_int_up() - result.get_mean()
			print  name.lower(), "\t\t= %3.3f" % result.get_mean(), "+/- %3.3f p=%3.3f" % (std, alpha)
			#weights = []
		
		#print "Kernel:", classifier.get_kernel()
		# add print output listener and mkl storage listener */
		#cross_validation.add_cross_validation_output(CrossValidationPrintOutput())
		#mkl_storage = CrossValidationMKLStorage()
		#cross_validation.add_cross_validation_output(mkl_storage)
		#result = cross_validation.evaluate()
		#weights=mkl_storage.get_mkl_weights()
		#print weights

		# ret.append( Result(metric=name.lower(), lower=result.get_conf_int_low(), upper=result.get_conf_int_up(), mean=result.get_mean(), alpha=result.get_conf_int_alpha(), weights=weights, crossval=[]) ) # Change this
		
	

def make_modelselection(param_tree_root, cross_validation, method='grid', ratio=0.5):
	if method == 'grid':
		return GridSearchModelSelection(param_tree_root, cross_validation)
	elif method == 'random':
		return RandomSearchModelSelection(param_tree_root, cross_validation, ratio)
	elif method == 'gradient':
		GradientModelSelection(param_tree_root, cross_validation)


def get_parameter_product(num_parameters, lower_range, upper_range):
	
	param_lists = [ xrange(lower_range, upper_range) for _ in xrange(num_parameters)]	

	return list(itertools.product(*param_lists))


def grid_search(objective):
	print "Hyper-parameter optimization with grid-search."
	best_objective = 0.0
	best_parameters = None
	history = {}

	parameter_tuple_order = 3 



	for p in get_parameter_product(parameter_tuple_order, -1, 10):

			current_parameters = map( lambda h: 2 ** float(h), p)
			objective_value = objective(current_parameters)
			history[ tuple(current_parameters) ] = objective_value

			if (objective_value > best_objective):
				best_objective = objective_value
				best_parameters = current_parameters

	return best_parameters, history


def random_search(objective):
	print "Hyper-parameter optimization with random-search."
	best_objective = 0.0
	best_parameters = None
	history = {}

	params = get_parameter_product(3, -2, 10)
	permut = map(int, np.random.permutation(len(params)))


	for idx in xrange(100):

			p = params[permut[idx]]
			current_parameters = map( lambda h: 2 ** float(h), p)
			objective_value = objective(current_parameters)
			history[ tuple(current_parameters) ] = objective_value

			if (objective_value > best_objective):
				best_objective = objective_value
				best_parameters = current_parameters

	return best_parameters, history



def gpo(objective):
	besto = 0.0
	bestp = None

	X = []
	y = []


	params = abs(rand(2)) * 10.0
	X.append(params)
	y.append(objective(params))

	params = abs(rand(2)) * 10.0
	X.append(params)
	y.append(objective(params))

	print "X = ", X
	print "y = ", y


	while(True):
		gp = GaussianProcess(theta0=0.001, thetaL=.001, thetaU=.002)
		gp.fit(X, y)   

		XX, YY = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 20))
		Z, mse = gp.predict(np.c_[XX.ravel(), YY.ravel()], eval_MSE=True)
		Z = np.array(Z)
		Z = Z.reshape(XX.shape)
		for n in X:
			print "Plotting: ", n
			pl.plot(n[0], n[1], '.r')
		
		CS = pl.contour(XX, YY, Z, 20, colours='k')
		pl.show()

		# Find next point to evaluate
		# Evaluate and append to X and y
		for k in xrange(30):
			params = abs(rand(2)) * 10.0
			X.append(params)
			y.append(objective(params))
		


	return bestp


def gpo1d(objective):
	besto = 0.0
	bestp = None

	D = 1
	

	X = []
	y = []


	params = abs(rand(D)) * 10.0
	X.append(params)
	y.append(objective([params, params]))

	params = abs(rand(D)) * 10.0
	X.append(params)
	y.append(objective([params, params]))

	print "X = ", X
	print "y = ", y


	while(True):
		gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
		gp.fit(X, y)   

		#XX, YY = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 20))
		#print XX

		XX = numpy.linspace(0, 10, 100)
		y_pred, mse = gp.predict(np.c_[XX], eval_MSE=True)
		sigma = np.sqrt(mse)
		#Z = np.array(Z)
		#Z = Z.reshape(XX.shape)
		
	
		pl.plot(X,y, 'xk')
		pl.plot(XX, y_pred, 'b-', label=u'Prediction')
		pl.fill(np.concatenate([XX, XX[::-1]]), np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]), alpha=.5, fc='b', ec='None', label='95% confidence interval')
		#CS = pl.contour(XX, YY, Z, 20, colours='k')
		pl.show()

		# Find next point to evaluate
		# Evaluate and append to X and y
		for k in xrange(2):
			params = abs(rand(D)) * 10.0
			X.append(params)
			y.append(objective([params, params]))
		


	return bestp


def load_configuration(path):
	print "Configuration ..."
	run = {}
	f = open(sys.argv[1])

	while True:
		line = f.readline().strip()
		if not line or line == "":
			break
		print line
		key, value = tuple(line.split("="))
		if key == "data":
			lst = run.get(key, [])
			lst.append(value)
			run[key] = lst
		else:
			run[key] = value
	print ""
	PARAM_BOUND = float(run.get("parameter-bound"))
	FOLDS       = int(run.get("folds"))
	RUNS        = int(run.get("runs"))
	DATASET     = run.get("data")
	METHOD      = run.get("parameter-selection")
	FEATURES    = int(run.get("feature-reduction"))
	return PARAM_BOUND, FOLDS, RUNS, DATASET, METHOD, FEATURES


""" Return a list of datasets, labels, ctype and K

Creates a synthetic list of datasets.

"""

def make_data(datasets, samples, features, informative):
	print "Loading data ... "
	data = []
	labs = []
	ctype = "MKL"
	K = datasets

	for i in xrange(datasets):
		# print "Making dataset", i
		X, y = make_classification(n_samples=samples)
		# X, y = make_classification(n_samples=samples, n_features=features, n_informative=informative, n_redundant=features-informative, n_repeated=0, n_classes=2, random_state=0)
		# print "Appending data"
		data.append(X.T)
		# print "Setting labs"
		labs = y
		r, c = X.nonzero()
		a, b = X.shape
		print " X =", X.shape, "\ty =", y.shape, "\tSparsity =", 1.0 * r.shape[0] / (a*b)

	for i in xrange(len(labs)):
		if (labs[i] == 0.0):
			labs[i] = -1.0

	return data, labs, ctype, K
