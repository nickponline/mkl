# cmaes.py

import numpy as np
import scipy as py
import pylab as pl

def rosen(x):  
	x = [x] if np.isscalar( x[0] ) else x  # scalar into list
	f = [sum(100.*(x[:-1]**2-x[1:])**2 + (1.-x[:-1])**2) for x in x]
	return f if len(f) > 1 else f[0]  # 1-element-list into scalar

N = 2
xmean = np.ones(N)
sigma = 0.5
stopfitness = 1e-10
stopeval = 1e3*N**2
print "N", N
print "xmean", xmean
print "sigma", sigma
print "stopfitness", stopfitness
print"stopeval", stopeval

h = 4+np.floor(3*np.log(N))
mu = h/2
weights = np.log(mu+1/2)-np.log(np.arange(1,mu)).T
mu = np.floor(mu)
weights = np.array(weights)/sum(weights)
mueff=sum(weights)**2 / sum(weights**2)
print "h", h
print "mu", mu
print "weighs", weights
print "mueff", mueff

cc = (4+mueff/N) / (N+4 + 2*mueff/N)
cs = (mueff+2) / (N+mueff+5)
c1 = 2 / ((N+1.3)**2+mueff)
cmu = 2 * (mueff-2+1/mueff) / ((N+2)**2+mueff)
damps = 1 + 2*max(0, np.sqrt((mueff-1)/(N+1))-1) + cs
print "cc", cc
print "cs", cs
print "c1", c1
print "cmu", cmu
print "damps", damps

pc = np.zeros((N,1))
B = np.eye(N,N)
D = np.ones((N,1))
C = B * np.diag(D**2) * B.T
invsqrtC = B * np.diag( D ) * B.T
eigeneval = 0
chiN=N**0.5*(1-1/(4*N)+1/(21*N**2))
print "pc", pc, pc.shape
print "B", B, B.shape
print "D", D, D.shape
print "C", C, C.shape
print "invsqrtC", invsqrtC, invsqrtC.shape
print "eigeneval", eigeneval
print "chiN", chiN, chiN
