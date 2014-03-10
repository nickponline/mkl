###########################################################################
# Mean prediction from Gaussian Processes based on 
# classifier_libsvm_minimal_modular.py
# plotting functions have been adapted from the pyGP library
# https://github.com/jameshensman/pyGP
###########################################################################
import pylab
import numpy as np
from numpy import *
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
from shogun.Regression import *
import pylab as PL
import matplotlib
import logging as LG
import scipy as SP
from shogun.ModelSelection import GradientModelSelection
from shogun.ModelSelection import ModelSelectionParameters, R_EXP, R_LINEAR
from shogun.ModelSelection import ParameterCombination


def run_demo():
    LG.basicConfig(level=LG.INFO)
    random.seed(572)

    x = np.linspace(0.0, 1.0, 80)
    y = np.sin(3.0 * np.pi * x)
    x = x[:, np.newaxis]

    feat_train   = RealFeatures(transpose(x));
    labels       = RegressionLabels(y);
    n_dimensions = 1 
        
    #new interface with likelihood parametres being decoupled from the covaraince function
    likelihood = GaussianLikelihood()
    covar_parms = SP.log([2])
    hyperparams = {'covar':covar_parms,'lik':SP.log([1])}
    
    #construct covariance function
    SECF  = GaussianKernel(feat_train, feat_train,2)
    covar = SECF
    zmean = ZeroMean();
    inf   = ExactInferenceMethod(SECF, feat_train, zmean, labels, likelihood);

    gp = GaussianProcessRegression(inf, feat_train, labels);
    
    root = ModelSelectionParameters();
    c1   = ModelSelectionParameters("inference_method", inf);
    root.append_child(c1);
    c2 = ModelSelectionParameters("scale");
    c1.append_child(c2);
    c2.build_values(0.01, 4.0, R_LINEAR);
    c3 = ModelSelectionParameters("likelihood_model", likelihood);
    c1.append_child(c3);
    c4 = ModelSelectionParameters("sigma");
    c3.append_child(c4);
    c4.build_values(0.001, 4.0, R_LINEAR);
    c5 = ModelSelectionParameters("kernel", SECF);
    c1.append_child(c5);
    c6 = ModelSelectionParameters("width");
    c5.append_child(c6);
    c6.build_values(0.001, 4.0, R_LINEAR);
    
    crit = GradientCriterion();

    grad = GradientEvaluation(gp, feat_train, labels, crit);
    grad.set_function(inf);

    gp.print_modsel_params();

    root.print_tree();

    grad_search=GradientModelSelection(root, grad)
    grad.set_autolock(0)

    best_combination=grad_search.select_model(1);

    x_test    = np.linspace(0.0, 1.0, 100)
    x_test    = x_test[:, np.newaxis]
    feat_test = RealFeatures(transpose(x_test));
    
    gp.set_return_type(GaussianProcessRegression.GP_RETURN_COV)

    St = gp.apply_regression(feat_test);   
    St = St.get_labels();
    
    gp.set_return_type(GaussianProcessRegression.GP_RETURN_MEANS);

    M = gp.apply_regression();
    M = M.get_labels();
    
    
    #create plots
    pylab.figure()
    # pylab.plot(x, y, 'rx')
    pylab.plot(x_test, M, 'ro')
    pylab.show()
    

run_demo()