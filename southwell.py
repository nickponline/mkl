import numpy as np
import scipy as sp
import pylab

from scipy.optimize import minimize


def rosen(x):
	"""The Rosenbrock function"""
	return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def objective(x):
	return sum(x ** 4)

def derivative(func, x, eps=1e-6):
	ret = np.array(x)
	for i,v in enumerate(x):
		forward = np.array(x)
		backward = np.array(x)
		forward[i] += eps
		backward[i] -= eps
		ret[i] = (func(forward) - func(backward)) / (2.0 * eps)
	return ret

def wolfe(func, x0, alpha):
	#print "x0, alpha = ", x0, alpha
	c1 = 1e-4
	c2 = 0.5

	partialDeriv = derivative(objective, x0)
	bestDirection = np.argmax(abs(partialDeriv))

	n = len(x0)
	bestVector = np.zeros(n)
	if partialDeriv[bestDirection] > 0.0:
		bestVector[bestDirection] = -1.0
	else:
		bestVector[bestDirection] = 1.0

	#print partialDeriv, bestVector

	a = None
	b = None
	
	if func(x0 + alpha * bestVector) <= (func(x0) + c1 * alpha * np.dot(bestVector,partialDeriv)):
		#print " Condition 1. True"
		a = True
	else:
		#print " Condition 1. False"
		a = False

	stepPartialDeriv = derivative(objective, x0 + alpha * bestVector)
	
	#print "Condition 2, left: ", abs(np.dot(partialDeriv, stepPartialDeriv)) 
	#print "Condition 2, right: " ,np.dot(bestVector, partialDeriv)
	#print "DOT 1: ", np.dot(bestVector, stepPartialDeriv)
	#print "DOT 2: ", np.dot(bestVector, partialDeriv) * c2

	if abs(np.dot(bestVector, stepPartialDeriv)) <= abs(c2 * np.dot(bestVector, partialDeriv)):
		#print " Condition 2. True"
		b = True
	else:
		#print " Condition 2. False"
		b = False

	#print "Wolfe conditions: ", a and b
	return a and b



def gauss_southwell(objective, x0, MAXITER=100):
	x0 = np.array(x0)
	n  = x0.shape
	v  = np.zeros(MAXITER)

	for k in xrange(MAXITER):
		j = 0
		tau = 0.1
		alpha = 1000.0

		while not wolfe(objective, x0, alpha) and j < 10:
			alpha = tau * alpha
			j = j + 1
			#print "Trying alpha of: ", alpha

		#print "Best alpha step is: ", alpha
		partialDeriv = derivative(objective, x0)
		bestDirection = np.argmax(abs(partialDeriv))

		bestVector = np.zeros(n)
		if partialDeriv[bestDirection] > 0.0:
			bestVector[bestDirection] = -1.0
		else:
			bestVector[bestDirection] = 1.0

		x0 = x0 + bestVector * alpha

		print "x0 => ", x0, "Obj => ", objective(x0)
		v[k] = objective(x0)
	return list(x0)

