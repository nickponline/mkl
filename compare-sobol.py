from sobol import *
from voronoi import *

import numpy.random
import pylab as pl
import random
import numpy
import ghalton

sequencer = ghalton.Halton(2)

iterations  = 1000
segmentSize = 1.0 / iterations
variableMin = -2.0
variableMax = 2.0

def obj(x):
	return numpy.exp(-x ** 2)

def latin_samples(iterations):
	ret = []
	for i in range(iterations):
	    segmentMin = i * segmentSize
	    point = segmentMin + (random.random() * segmentSize)
	    ret.append(point)
	random.shuffle(ret)
	return numpy.array(ret)

def toVor(X, Y):
	Z = zip(X.T.ravel(), Y.T.ravel())
	Z = map(list, Z)
	Z = numpy.array(Z)
	return Z

def halton(iterations):
	H = numpy.array(sequencer.get(iterations))
	return H[:, 0], H[:, 1]

def describe(X, Y):
	distances = []
	for i in xrange(1, iterations):
		dsb = 1000000.0
		for k in xrange(0, i):
			ds = ((X.ravel()[i] - X.ravel()[i-1]) ** 2) + ((Y.ravel()[i] - Y.ravel()[i-1]) ** 2)
			if ds < dsb:
				dsb = ds	
		distances.append(numpy.sqrt(dsb))
	return numpy.mean(distances), numpy.std(distances)

Xu, Yu = numpy.random.random((1, iterations)), numpy.random.random((1, iterations))
Xs, Ys = i4_sobol_generate(2, iterations, 0)
Xl, Yl = latin_samples(iterations), latin_samples(iterations)
Xh, Yh = halton(iterations)


print describe(Xu, Yu)
print describe(Xl, Yl)
print describe(Xs, Ys)
print describe(Xh, Yh)

# voronoi(toVor(Xu, Yu))
# voronoi(toVor(Xs, Ys))
# voronoi(toVor(Xl, Yl))
# voronoi(toVor(Xh, Yh))


# pl.figure()
# pl.subplot(3, 1, 1)
# pl.plot(Xu, Yu, 'k+')
# pl.subplot(3, 1, 2)
# pl.plot(Xs, Ys, 'k+')
# pl.subplot(3, 1, 3)
# pl.plot(Xl, Yl, 'k+')

# t = numpy.linspace(variableMin, variableMax, 100)
# f = obj(t)

# t_uniform = variableMin + (variableMax - variableMin) * numpy.random.random((1, iterations))
# f_uniform = obj(t_uniform)

# t_sobol = variableMin + (variableMax - variableMin) * i4_sobol_generate(1, iterations, 0)
# f_sobol = obj(t_sobol)

# t_latin = variableMin + (variableMax - variableMin) * latin_samples(iterations)
# f_latin = obj(t_latin)

# pl.figure()
# pl.subplot(1, 3, 1)
# pl.plot(t, f)
# pl.plot(t_uniform, f_uniform, 'k+')

# pl.subplot(1, 3, 2)
# pl.plot(t, f)
# pl.plot(t_sobol, f_sobol, 'k+')

# pl.subplot(1, 3, 3)
# pl.plot(t, f)
# pl.plot(t_latin, f_latin, 'k+')

# pl.show()

