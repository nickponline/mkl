import pickle
import numpy
import scipy
import scipy.ndimage
import pylab as pl

#X1 = pickle.load(open( "X1.p", "rb" ) )
#X2 = pickle.load(open( "X2.p", "rb" ) )
#XX = pickle.load(open( "XX.p", "rb" ) )
#YY = pickle.load(open( "YY.p", "rb" ) )
Z  = pickle.load(open( "ZZ.p", "rb" ) )
Z = scipy.ndimage.zoom(Z, 7)
CS = pl.contour(Z, 30, colors='k')
pl.clabel(CS, fontsize=9, inline=1)
pl.title('Single color - negative contours dashed')
pl.show()