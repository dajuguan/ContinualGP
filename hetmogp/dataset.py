import numpy as np
import sys

sys.path.append("..")

import GPy

from likelihoods.bernoulli import Bernoulli
from likelihoods.gaussian import Gaussian
from likelihoods.hetgaussian import HetGaussian
from likelihoods.beta import Beta
from likelihoods.gamma import Gamma
from likelihoods.exponential import Exponential
from likelihoods.poisson import Poisson
from hetmogp.het_likelihood import HetLikelihood

def load_toy2(N=500,input_dim=2):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)

    # Heterogeneous Likelihood Definition
    likelihoods_list = [HetGaussian(), Beta(), Bernoulli(), Gamma(), Exponential()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 3
    """"""""""""""""""""""""""""""

    Dim = input_dim
    if input_dim ==2:
        xy = np.linspace(0.0, 1.0, Nsqrt)
        xx = np.linspace(0.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = 0 * np.ones(Dim)
        maxis = 1 * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            lenghtscale = np.array([0.5, 0.05, 0.1]) #This is the one used for previous experiments
        else:
            lenghtscale = lenghtscale

        if variance is None:
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):
        #true_f = []

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        # q=1
        for q in range(Q):
            if q==0:
                W_list.append(np.array([-0.1,-0.1, 1.1, 2.1, -1.1, -0.5, -0.6, 0.1])[:,None])
            elif q == 1:
                W_list.append(np.array([1.4, -0.5, 0.3, 0.7, 1.5, -0.3, 0.4, -0.2])[:,None])
            else:
                W_list.append(np.array([0.1, -0.8, 1.3, 1.5, 0.5,-0.02, 0.01, 0.5])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Yreg = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    Ytrain = [Yreg,np.clip(Ytrain[1],1.0e-9,0.99999),Ytrain[2],Ytrain[3],Ytrain[4]]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain