import sys
import numpy
import pylab
import pickle
import pickle
import itertools
import scipy.sparse
import scipy.ndimage
import scipy.stats

from time import clock, time

from random import *
from numpy import *
from numpy.random import randn

from scipy.optimize import *
from scipy.sparse import *

from shogun.Classifier import *
from shogun.Mathematics import *
from shogun.Kernel import *
from shogun.Features import *
from shogun.Evaluation import *

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.gaussian_process import GaussianProcess
from sklearn.datasets import make_classification

from sobol import *
from pylab import *
from collections import namedtuple
import ghalton

from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

PARAMETERS = {}

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


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
		#print X, y, g, notes
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



def make_features(data, kernel_type):
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
		if kernel_type == "GaussianSpread":
			for i in xrange(10):
				for dt in data:
					sub = None
					if isinstance(dt, scipy.sparse.csc_matrix):
						sub = SparseRealFeatures(dt)
					elif isinstance(dt, numpy.ndarray):
						sub = RealFeatures(dt)
					features.append_feature_obj(sub)
					
		else:			
			for dt in data:
				sub = None
				if isinstance(dt, scipy.sparse.csc_matrix):
					sub = SparseRealFeatures(dt)
				elif isinstance(dt, numpy.ndarray):
					sub = RealFeatures(dt)
				features.append_feature_obj(sub)
		return features;

def make_kernel(K, features, kernel_type, ctype, widths):
	# Single kernel
	#print "Here"
	assert(K == len(widths))

	if ctype=="SKL":
		if kernel_type == "GaussianKernel":
			kernel=GaussianKernel()	
		if kernel_type == "LinearKernel":
			kernel=LinearKernel()	
		if kernel_type == "GaussianARDKernel":
			kernel=GaussianARDKernel()	
		if kernel_type == "GaussianSpread":
			kernel = GaussianKernel()
		
		#kernel.set_normalizer(TanimotoKernelNormalizer(1.2))
		kernel.init(features, features)
		#kernel.print_modsel_params()
		return kernel
	# Multiple kernel
	elif ctype=="MKL":
		kernel = CombinedKernel()
		for i in xrange(K): # 5 kernels
			if kernel_type == "GaussianKernel":
				subkernel=GaussianKernel()	
				kernel.append_kernel(subkernel)
			if kernel_type == "LinearKernel":
				subkernel=LinearKernel()	
				kernel.append_kernel(subkernel)
			if kernel_type == "GaussianARDKernel":
				subkernel=GaussianARDKernel()	
				kernel.append_kernel(subkernel)
			if kernel_type == "GaussianSpread":
				for w in xrange(10):
					
					subkernel=GaussianKernel()	
					subkernel.set_width(np.power(2.0, w))
					kernel.append_kernel(subkernel)
			
			# subkernel.set_normalizer(TanimotoKernelNormalizer())
			
		
		#print kernel.get_kernel_matrix()
		kernel.init(features, features)
		print "\n".join(dir(kernel))
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

def evaluate(bestVector, kernel, kernel_type, ctype, classifier, train_labels, train_features, folds, runs, alpha = 0.05):


	c = abs(bestVector[0])
	w = map(abs, bestVector[1:])
	
	classifier.set_C(c, c)

	set_kernel_parameters(kernel, w, kernel_type)
	
	headers = []
	ret = []
	results = {}
	for code, name in [(0, "ACCURACY"), (10, "ERROR_RATE"), (20, "BAL"), (30, "WRACC"), (40, "F1"), (50, "CROSS_CORRELATION"), (60, "RECALL"), (70, "PRECISION"), (80, "SPECIFICITY")]:
	# for code, name in [(40, "F1")]:
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
			A = name.capitalize()
			B = name.capitalize() + "MeanWeights"
			C = name.capitalize() + "StdWeights"
			print  name.lower(), "\t\t= %3.3f" % result.get_mean(), "+/- %3.3f p=%3.3f" % (std, alpha)
			print "mean-weights     =" , map(lambda h: "%3.3f" % h, Statistics.matrix_mean(weights, False))
			# print "variance-weights =" , map(lambda h: "%3.3f" % h, Statistics.matrix_variance(weights, False))
			print "std-weights      =" , map(lambda h: "%3.3f" % h, Statistics.matrix_std_deviation(weights, False))
			results[A] = [result.get_mean(), std, alpha]
			results[B] = map(lambda h: "%3.3f" % h, Statistics.matrix_mean(weights, False))
			results[C] = map(lambda h: "%3.3f" % h, Statistics.matrix_std_deviation(weights, False))
			

		else:
			#cross_val_output = CrossValidationPrintOutput()
			#cross_validation.add_cross_validation_output(cross_val_output)
			result  = cross_validation.evaluate()
			std = result.get_conf_int_up() - result.get_mean()
			print  name.lower(), "\t\t= %3.3f" % result.get_mean(), "+/- %3.3f p=%3.3f" % (std, alpha)
			A = name.capitalize()
			results[A] = [result.get_mean(), std, alpha]
			#weights = []
		
		#print "Kernel:", classifier.get_kernel()
		# add print output listener and mkl storage listener */
		#cross_validation.add_cross_validation_output(CrossValidationPrintOutput())
		#mkl_storage = CrossValidationMKLStorage()
		#cross_validation.add_cross_validation_output(mkl_storage)
		#result = cross_validation.evaluate()
		#weights=mkl_storage.get_mkl_weights()
		#print weights
	return results
		# ret.append( Result(metric=name.lower(), lower=result.get_conf_int_low(), upper=result.get_conf_int_up(), mean=result.get_mean(), alpha=result.get_conf_int_alpha(), weights=weights, crossval=[]) ) # Change this
		
	

def make_modelselection(param_tree_root, cross_validation, method='grid', ratio=0.5):
	if method == 'grid':
		return GridSearchModelSelection(param_tree_root, cross_validation)
	elif method == 'random':
		return RandomSearchModelSelection(param_tree_root, cross_validation, ratio)
	elif method == 'gradient':
		GradientModelSelection(param_tree_root, cross_validation)



def get_parameter_product(num_parameters, lower_range, upper_range, steps=15):
	
	param_lists = [ numpy.linspace(lower_range, upper_range, steps) for _ in xrange(num_parameters)]	

	return (itertools.product(*param_lists))


def parameter_gradient(objective, params, episilon=0.000001):
    N = len(params)
    
    gradient_vector = []
    
    for i in xrange(N):
    	params_forward   = [[params[i],params[i]+episilon][i == x] for x in xrange(N)]
    	params_backward  = [[params[i],params[i]-episilon][i == x] for x in xrange(N)]
    	print params_forward
    	print params_backward

        gradient_vector.append( objective(params_forward) - objective(params_backward) / 2.0*episilon )

    return gradient_vector


def grid_search(objective, PARAMETERS, PARAM_BOUND):
	
	print "Hyper-parameter optimization with grid-search."
	best_objective = 0.0
	best_parameters = None
	history = {}
	grid = []

	parameter_tuple_order = int(PARAMETERS) + 1

	for p in get_parameter_product(parameter_tuple_order, -int(PARAM_BOUND), int(PARAM_BOUND)):
			
			current_parameters = map( lambda h: 2 ** float(h), p)
			# print "Gradients at:", current_parameters
			# print parameter_gradient(objective, current_parameters)
			objective_value = objective(current_parameters)
			history[ tuple(current_parameters) ] = objective_value
			grid.append(objective_value)
			if (objective_value > best_objective):
				best_objective = objective_value
				best_parameters = current_parameters

	return best_parameters, history, grid


def random_search(objective, PARAMETERS, PARAM_BOUND):
	print "Hyper-parameter optimization with random-search."
	best_objective = 0.0
	best_parameters = None
	history = {}
	grid = []

	lookup = [x for x in xrange( -int(PARAM_BOUND), int(PARAM_BOUND))]
	
	startTime = time()
	for idx in xrange(2048):

			p = [ choice(lookup) for x in xrange(2 + int(PARAMETERS))]
			current_parameters = map( lambda h: 2 ** float(h), p)
			objective_value = objective(current_parameters)
			history[ tuple(current_parameters) ] = objective_value

			if (objective_value > best_objective):
				best_objective = objective_value
				best_parameters = current_parameters


			if is_power2(idx+1):  
				endTime = time()
				history = {}
				objective(best_parameters, [0, 20, 50, 40], production=True)
				print idx+1, " & ", idx+1, " & ", int(endTime) - int(startTime), "\\\\"
				results = {}

	return best_parameters, history, grid

def sobol_search(objective, PARAMETERS, PARAM_BOUND):
	print "Hyper-parameter optimization with random-search."
	best_objective = 0.0
	best_parameters = None
	history = {}
	grid = []

	lookup = [x for x in xrange( -int(PARAM_BOUND), int(PARAM_BOUND))]
	
	startTime = time()

	sd = 17
	
	for idx in xrange(2048):

			p = [ choice(lookup) for x in xrange(2 + int(PARAMETERS))]
			s, sd = i4_sobol(2 + int(PARAMETERS), sd)
			p = s * 10.0

			current_parameters = map( lambda h: 2 ** float(h), p)
			objective_value = objective(current_parameters)
			history[ tuple(current_parameters) ] = objective_value

			if (objective_value > best_objective):
				best_objective = objective_value
				best_parameters = current_parameters


			if is_power2(idx+1):  
				endTime = time()
				history = {}
				objective(best_parameters, [0, 20, 50, 40], production=True)
				print idx+1, " & ", idx+1, " & ", int(endTime) - int(startTime), "\\\\"
				results = {}

	return best_parameters, history, grid

def latin_samples(iterations, dimensions):
	segmentSize = 1.0 / iterations
	variableMin = -2.0 ** 15.0
	variableMax =  2.0 ** 15.0
	ret = []
	rett = []
	for k in xrange(dimensions):
		for i in range(iterations):
		    segmentMin = i * segmentSize
		    point = segmentMin + (random() * segmentSize)
		    ret.append(point)
		shuffle(ret)
		rett.append(ret[0])
	return rett


def halton(sequencer, iterations, dimensions):
	H = numpy.array(sequencer.get(iterations))
	return H[:, 0], H[:, 1]

def latin_search(objective, PARAMETERS, PARAM_BOUND):
	print "Hyper-parameter optimization with LHS."
	best_objective = 0.0
	best_parameters = None
	history = {}
	grid = []

	lookup = [x for x in xrange( -int(PARAM_BOUND), int(PARAM_BOUND))]
	
	startTime = time()

	sd = 17
	
	for idx in xrange(2048):

			p = [ choice(lookup) for x in xrange(2 + int(PARAMETERS))]
			s = latin_samples(1, 2 + int(PARAMETERS))
			p = s

			current_parameters = p
			objective_value = objective(current_parameters)
			history[ tuple(current_parameters) ] = objective_value

			if (objective_value > best_objective):
				best_objective = objective_value
				best_parameters = current_parameters


			if is_power2(idx+1):  
				endTime = time()
				history = {}
				objective(best_parameters, [0, 20, 50, 40], production=True)
				print idx+1, " & ", idx+1, " & ", int(endTime) - int(startTime), "\\\\"
				results = {}

	return best_parameters, history, grid

def halton_search(objective, PARAMETERS, PARAM_BOUND):
	print "Hyper-parameter optimization with Halton."
	best_objective = 0.0
	best_parameters = None
	history = {}
	grid = []

	lookup = [x for x in xrange( -int(PARAM_BOUND), int(PARAM_BOUND))]
	
	startTime = time()

	sd = 17
	sequencer = ghalton.Halton(2 + int(PARAMETERS))
	
	
	for idx in xrange(2048):

			p = [ choice(lookup) for x in xrange(2 + int(PARAMETERS))]
			s = halton(sequencer, 1, 2 + int(PARAMETERS))
			#print s
			p = s

			current_parameters = p
			objective_value = objective(current_parameters)
			#history[ tuple(current_parameters) ] = objective_value

			if (objective_value > best_objective):
				best_objective = objective_value
				best_parameters = current_parameters


			if is_power2(idx+1):  
				endTime = time()
				history = {}
				objective(best_parameters, [0, 20, 50, 40], production=True)
				print idx+1, " & ", idx+1, " & ", int(endTime) - int(startTime), "\\\\"
				results = {}

	return best_parameters, history, grid


def fun(x):
	d1 = (0.25 - x)
	d2 = (0.75 - x)
	return np.exp(-d1*d1 / 0.05) * 1.5 + np.exp(-d2*d2 / 0.05) * 2.7

def sqr(x):
	return x*x

def fun_2d(x):
	d1 = sqr(2.25 - x[0]) + sqr(2.25 - x[1])
	d2 = sqr(6.75 - x[0]) + sqr(6.75 - x[1])
	d3 = sqr(5.25 - x[0]) + sqr(2.25 - x[1])
	d4 = sqr(2.75 - x[0]) + sqr(5.75 - x[1])
	
	return np.exp(-d1 / 16.0) * 0.0 + np.exp(-d2 / 16.0) * 4.7

def fun_5d(x):
	d1 = sqr(2.25 - x[0]) + sqr(2.25 - x[0])
	d2 = sqr(6.75 - x[1]) + sqr(6.75 - x[1])
	d3 = sqr(5.25 - x[2]) + sqr(2.25 - x[2])
	d4 = sqr(2.75 - x[3]) + sqr(5.75 - x[3])
	d5 = sqr(2.75 - x[4]) + sqr(5.75 - x[4])
	
	return np.exp(-d1 + d2 + d3 / 16.0) * 0.0 + np.exp(-d4 + d5 / 16.0) * 4.7


def gaussian_process_surrogate_2d(objective, search='ei'):
	besto = 0.0
	bestp = None

	dimensions = 2

	X = []
	y = []

	for k in xrange(10):	
		params = abs(rand(dimensions)) * 10.0
		X.append(params)
		y.append(objective(params))

	

	figa = pylab.figure()
	figb = pylab.figure()
	figc = pylab.figure()
	
	subx = 3
	suby = 3
	index = 1

	while(index <= subx*suby):
		# print plotindex
		# if plotindex > plotsx * plotsy:
		# 	break

		gp = GaussianProcess(theta0=0.001, thetaL=.001, thetaU=.002)
		gp.fit(X, y)   

		Xr = np.linspace(0, 10, 100)
		Yr = np.linspace(0, 10, 100)

		XX, YY = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
		Z, mse = gp.predict(np.c_[XX.ravel(), YY.ravel()], eval_MSE=True)
		y_pred = Z
		Z = np.array(Z)
		Z = Z.reshape(XX.shape)

		axa = figa.add_subplot(subx, suby, index)
		axb = figb.add_subplot(subx, suby, index)
		axc = figc.add_subplot(subx, suby, index)
		
		index = index + 1
		
		for n in X:
			# print "Plotting: ", n
			axa.plot(n[0], n[1], 'or')
			axb.plot(n[0], n[1], 'or')
			axc.plot(n[0], n[1], 'or')

		# Find next point to evaluate
		# Evaluate and append to X and y

		sigma = np.sqrt(mse)

		fbest = np.array(y).max()
		u = (y_pred - fbest) / sigma

		if search == "ei":
			ei = sigma * (u * scipy.stats.norm.cdf(u) + scipy.stats.norm.pdf(u))
		else:
			T = fbest + (0.05 * numpy.abs(fbest))
			ei = 1.0 - scipy.stats.norm.cdf((T - y_pred) / sigma)	

		bestindex = np.argmax(ei)
		bx, by = XX.ravel()[bestindex], YY.ravel()[bestindex]

		print search, ei[bestindex], "at", bx, by
		ei_square = ei.reshape(XX.shape)

		CS = axa.contourf(XX, YY, Z)
		CS = axb.contourf(XX, YY, fun_2d([XX,YY]))
		CS = axc.contourf(XX, YY, ei_square)

		# subplot(2, 1, 2)

		# for n in X:
		# 	# print "Plotting: ", n
		# 	pylab.plot(n[0], n[1], 'ok')
		
		# CS = pylab.contour(XX, YY, ei_square, 20, colours='k')
		
	
		params = [bx, by]

		skip = False
		for p in X:
			if p[0] == bx and p[1] == by:
				skip = True

		if not skip:
			X.append(params)
			y.append(objective(params))
	
	print "Saving"		
	figa.savefig("2da%s.eps" % search)
	figb.savefig("2db%s.eps" % search)
	figc.savefig("2dc%s.eps" % search)
	
	pylab.show()
	return bestp



def add_indices(indx, lower, upper):
	done = False
	while not done:
		ran = randint(lower, upper)
		
		try:
			idx = indx.index(ran)
			
			done = False
		except:
			indx.append(ran)
			
			done = True
	
	return indx

def fun(x):
	d1 = (0.25 - x)
	d2 = (0.75 - x)
	return np.exp(-d1*d1 / 0.05) * 1.5 + np.exp(-d2*d2 / 0.05) * 2.7

def gaussian_process_surrogate(objective, search="ei"):

	besto = 0.0
	bestp = None

	D = 1
	
	indx = []


	tx = np.linspace(0.0, 1.0, 1000)
	ty = fun(tx)
		
	indx = add_indices(indx, 0, 999)
	indx = add_indices(indx, 0, 999)
	
	plotindex = 1
	plotsx = 3
	plotsy = 3
	
	pylab.figure(figsize=(10, 10))


	while(True):

		if plotindex > plotsx * plotsy:
			break

		X = np.array([tx[indx]]).T
		y = (fun(X)).ravel()

		bestp = y.max()
		
		print indx
		gp = GaussianProcess(corr='squared_exponential')	
		gp.fit(X, y)   


		XX = numpy.linspace(0.0, 1.0, 1000)
		y_pred, mse = gp.predict(np.c_[XX], eval_MSE=True)
		sigma = np.sqrt(mse)

		fbest = y.max()
		u = (y_pred - fbest) / sigma

		if search == "ei":
			ei = sigma * (u * scipy.stats.norm.cdf(u) + scipy.stats.norm.pdf(u))
		else:
			T = fbest + (0.25 * numpy.abs(fbest))
			ei = 1.0 - scipy.stats.norm.cdf((T - y_pred) / sigma)	

		bestnext = np.argmax(ei)

		try:
			indx.index(bestnext)
			indx = add_indices(indx, 0, 99)
		except:
			indx.append(bestnext)

		offset = 1.0
		subplot(plotsx, plotsy, plotindex)
		pylab.plot([tx[bestnext], tx[bestnext]], [0.0, y_pred[bestnext] +offset], ':b')
		pylab.plot(tx, ty  + offset, 'k:')
		pylab.plot(X,y  + offset, 'or')
		pylab.plot(XX, y_pred + offset, 'b-', label=u'Prediction')
		pylab.fill(np.concatenate([XX, XX[::-1]]), offset + np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]), alpha=.1, fc='b', ec='None', label='95% confidence interval')
		pylab.plot(XX, ei / max(ei.max(), 0.000001) * 1.0, 'k-')
		pylab.xlabel(r"$\mathbf{x}$")
		if search == "ei":
			pylab.ylabel(r"$\mathrm{EI(\mathbf{x})}$")
		else:
			pylab.ylabel(r"$\mathrm{PI(\mathbf{x})}$")
		# pylab.legend(['True Performance', 'Sampled Performance', 'Expected Improvement'], loc='best' )
		pylab.axis([0, 1, 0, 5])
		plotindex = plotindex + 1
	
	pylab.savefig(search + ".eps", figsize=(10, 10))
	pylab.show()
	return bestp

def myr(lower, upper):
	return lower + random()*(upper - lower)

def subobjective(x, gp, fbest, search='ei'):
	#print "Evaluating subobjective: ",  x
	try:
		y_pred, mse = gp.predict(x, eval_MSE=True)
		sigma = np.sqrt(mse)

		if sigma == 0.0:
			sigma = 0.00001

		u = (y_pred - fbest) / sigma
		val = None


		#print "Computing criterion ...."
		if search == "ei":
			val = sigma * (u * scipy.stats.norm.cdf(u) + scipy.stats.norm.pdf(u))
		else:
			T = fbest + (0.25 * numpy.abs(fbest))
			val = 1.0 - scipy.stats.norm.cdf((T - y_pred) / sigma)	
	except:
		return 0.0

	return val

def random_vector(dimensions, lower, upper):
	return np.array([myr(lower, upper) for i in xrange(dimensions)])

def gaussian_process_surrogate_nd(objective, dimensions, search="ei"):

	besto = 0.0
	bestp = None

	lower      = 2.0 ** -15
	upper      = 2.0 ** 15
	
	indx = []

	#tx = np.linspace(lower, upper, 1000)
	#ty = objective(tx)


	
	X = []
	y = []

	H = ghalton.Halton(dimensions)
	halton = numpy.array(H.get(10 * dimensions))
	print "Halton Size: ", halton.shape

	seed = 34
	

	#L = numpy.hstack( [ latin_samples(10 * dimensions) for _ in xrange(dimensions) ] )
	#print "LHS Size: ", L.shape

	print "Warming up ..."
	for i in xrange(10 * dimensions):
		
		# Uniform
		#r = random_vector(dimensions, lower, upper)
		
		# Halton
		#r = halton[i, :]

		# Sobol
		r, seed = i4_sobol(dimensions, seed)


		X.append(r)
		y.append(objective(r))

	print "Done"
	
	
	#pylab.figure(figsize=(10, 10))

	beste = 0.0

	for _ in xrange(100):


		gp = GaussianProcess(corr='squared_exponential')
		#gp = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, verbose=True, optimizer='Welch')	
		try:
			gp.fit(np.c_[X], np.c_[y])   
		except:
			print "Skipping."
		
		

		# XX = numpy.linspace(lower, upper, 1000)
		# y_pred, mse = gp.predict(np.c_[XX], eval_MSE=True)
		# sigma = np.sqrt(mse)

		# fbest = np.array(y).max()
		# u = (y_pred - fbest) / sigma

		# if search == "ei":
		# 	ei = sigma * (u * scipy.stats.norm.cdf(u) + scipy.stats.norm.pdf(u))
		# else:
		# 	T = fbest + (0.25 * numpy.abs(fbest))
		# 	ei = 1.0 - scipy.stats.norm.cdf((T - y_pred) / sigma)	

		#print "Minimizing ..."
		fbest          = np.array(y).max()
		start_location = random_vector(dimensions, lower, upper)
		#print "Start location: ", start_location
		#bestnext = minimize_scalar(lambda f: -subobjective(f, gp, fbest), bounds=(lower, upper), method='bounded')
		bestnext = minimize(lambda f: -subobjective(f, gp, fbest), start_location, bounds=[(lower, upper) for _ in xrange(dimensions)], method='L-BFGS-B')
		bestnext = bestnext.x
		#print "Best minimizer: ", bestnext
		
		X.append(bestnext)
		e = objective(bestnext)
		if e > beste:
			beste = e
			print "Best accuracy:", beste
		y.append(e)

		# offset = 1.0
		
		# subplot(plotsx, plotsy, plotindex)
		
		# # Objective function
		# pylab.plot(tx, ty + offset, 'k:')
		# # Best criterion
		# pylab.plot([bestnext, bestnext], [0.0, objective(bestnext) + offset], ':b')
		
		# # Plot evaluation locations
		# pylab.plot(np.array(X),np.array(y)  + offset, 'or')

		# # Plot predictive mean 
		# pylab.plot(XX, y_pred + offset, 'b-', label=u'Prediction')
		# #pylab.fill(np.concatenate([XX, XX[::-1]]), offset + np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]), alpha=.1, fc='b', ec='None', label='95% confidence interval')
		
		# pylab.plot(XX, subobjective(XX, gp, fbest,  search='ei'), 'r')
		
		# pylab.axis([lower, upper, 0, 5])


	index = numpy.array(y).argmin(axis=0)
	#print "Solution: ", X[index]
	return X[index]


def load_configuration(path):
	print "Configuration ..."
	run = {}
	f = open(sys.argv[1])

	lines = f.read().strip().split("\n")

	for line in lines:
		if line == "":
			continue
		
		print line
		key, value = tuple(line.split("="))
		if key == "DATA":
			lst = run.get(key, [])
			lst.append(value)
			run[key] = lst
		else:
			run[key] = value
	print ""

	return run

def raw_configuration(path):
	print "Configuration ..."
	run = {}
	f = open(sys.argv[1])
	return f.read()



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

def set_inner_kernel_parameters(kernel, w):

    # Only current consider Gaussian and linear kernels
    if kernel.get_name() == "GaussianKernel":
        kernel.set_width(w[0])
    elif kernel.get_name() == "WeightedDegreeRBFKernel":
        kernel.set_width(w)
    elif kernel.get_name() == "GaussianARDKernel":
        for i in xrange(len(w)):
            kernel.set_weight(w[i], i)
    else:
        pass


def set_kernel_parameters(kernel, w, kernel_type):

	if kernel.get_name() == "CombinedKernel":
		for i in xrange(kernel.get_num_subkernels()): 
			if kernel.get_kernel(i).get_name() == "GaussianKernel":
				gaussian_kernel = GaussianKernel.obtain_from_generic(kernel.get_kernel(i))
				if kernel_type != "GaussianSpread":
					if len(w) == 1:
						gaussian_kernel.set_width(float(w[0]))
					else:
						gaussian_kernel.set_width(float(w[i]))

			elif kernel.get_kernel(i).get_name() == "GaussianARDKernel":
				gaussian_kernel = GaussianARDKernel.obtain_from_generic(kernel.get_kernel(i))
				idx = 0
				for p in xrange(gaussian_kernel.num_features()):
					gaussian_kernel.set_width(w[idx], p)
					idx = idx + 1
			else:
				pass
	else:
		set_inner_kernel_parameters(kernel, w)


def describe_kernel(kernel):

    if kernel.get_name() == "CombinedKernel":
        print "Combined Kernel"
        for i in xrange(kernel.get_num_subkernels()): 
            print i, kernel.get_kernel(i).get_name()
    else:
        print kernel.get_name()
    print ""
