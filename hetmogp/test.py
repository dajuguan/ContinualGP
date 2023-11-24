import sys
import numpy as np
import random
sys.path.append("..")

from hetmogp.svmogp import SVMOGP
from hetmogp import util
from likelihoods.gaussian import Gaussian
from hetmogp.het_likelihood import HetLikelihood
from hetmogp.util import vem_algorithm as VEM
# from hetmogp.dataset import load_toy2

def multi_output():
    # Xtrain, Ytrain = load_toy2()
    n = 50
    x1 = np.linspace(1,2,n) * 10
    x2 = np.linspace(1,2,n) * 10
    x = np.stack((x1,x2), axis=-1)
    # xs, ys = np.meshgrid(x,y,sparse=True)
    y1 = x1 **2 + x2**2
    y1.resize(n,1)
    y2 = x1 + x2
    y2.resize(n,1)
    Xtrain = [x,x]
    Ytrain = [y1,y2]
    num_inducing = 40
    Q = 2
    VEM_its = 20

    random.seed(10)   #We use this seed to guarantee all inner variables of the model be fairly started for methods

    likelihoods_list = [Gaussian(0.001), Gaussian(0.001)]
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()

    minis = Xtrain[0].min(0)
    maxis = Xtrain[0].max(0)
    Dim = Xtrain[0].shape[1]
    print("dim is:", Dim)

    Z = np.linspace(minis[0],maxis[0],num_inducing).reshape(1,-1)
    for i in range(Dim-1):
        Zaux = np.linspace(minis[i+1],maxis[i+1],num_inducing)
        Z = np.concatenate((Z,Zaux[np.random.permutation(num_inducing)].reshape(1,-1)),axis=0)
        #Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
    Z = 1.0*Z.T
    # Z = x

    # ls_q = np.sqrt(Dim) * (np.random.rand(Q) + 0.05)
    # var_q = 0.05 * np.ones(Q)  
    ls_q = np.array(([.05] * Q))
    var_q = 0.05 * np.ones(Q)  
    # kern_list = util.latent_functions_prior(Q, lenghtscale=ls_q, variance=var_q, input_dim=Dim)
    kern_list = util.latent_functions_prior(Q, input_dim=Dim)

    print("-----------------------------------")

    """""""""""""""""""""""""""""""""Creating the HeMOGP Model"""""""""""""""""""""""""""""""""
    model = SVMOGP(X=Xtrain, Y=Ytrain, Z=Z.copy(), kern_list=kern_list, likelihood=likelihood, Y_metadata=Y_metadata,batch_size=None)

    # model['.*.kappa'].fix()
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #Random initialisation of the linear combination coefficients 
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # for q in range(Q):
    #     model['B_q' + str(q) + '.W'] = 2.0 * np.random.rand(model['B_q0.W'].__len__())[:, None]
        # model.kern_list[q].variance.fix()
        # model.kern_list[q].variance = 1.0e-8
        # model.kern_list[q].white.variance = 1.0e-8
        # model.kern_list[q].white.fix()

    hetmogp_model = VEM(model, stochastic=False, vem_iters=VEM_its, optZ=False, verbose=False, verbose_plot=False, non_chained=False)
    X_new = np.array([[0,0],[0.2,0.2]])
    mu_0, _ = hetmogp_model._raw_predict_f(X_new)
    print("mu_0:", mu_0)
    mu_1, _ = hetmogp_model._raw_predict_f(X_new, 1)
    print("mu_1:", mu_1)

multi_output()