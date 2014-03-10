import numpy as np
from sklearn.gaussian_process import GaussianProcess
import pylab as pl
import pickle
from numpy import exp, log

X1 = pickle.load(open( "X1.p", "rb" ) )
X2 = pickle.load(open( "X2.p", "rb" ) )
XX = pickle.load(open( "XX.p", "rb" ) )
YY = pickle.load(open( "YY.p", "rb" ) )
Z  = pickle.load(open( "ZZ.p", "rb" ) )
Z  = np.array(Z)
A = np.array(X1).ravel()
B = np.array(X2).ravel()



X = np.array([exp(XX.ravel()),exp(YY.ravel())]).T
y = np.array(Z).ravel() # .T ?
print X.shape
print y.shape
gp = GaussianProcess(theta0=0.001, thetaL=.001, thetaU=.002)
print X.shape
print y.shape
gp.fit(X, y)                                      

X1, X2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
print X1.shape
print X2.shape
Z, mse = gp.predict(np.c_[X1.ravel(), X2.ravel()], eval_MSE=True)

print Z.shape

Z = Z.reshape(X1.shape)
#pl.pcolormesh(X1, X2, Z, cmap=pl.cm.Paired)
CS = pl.contour(X1, X2, Z, 10)
pl.plot(A, B, '.r')
pl.clabel(CS, fontsize=9, inline=1)
pl.title('Single color - negative contours dashed')

pl.show()