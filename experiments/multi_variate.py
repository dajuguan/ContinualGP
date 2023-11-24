import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

sys.path.append("..")

from continualgp.het_likelihood import HetLikelihood
from continualgp.continualgp import ContinualGP
from continualgp import util
from likelihoods.gaussian import Gaussian
from continualgp.util import vem_algorithm as onlineVEM

Q = 1  # number of latent functions
VEM_its = 10

n = 20
x1 = np.linspace(0,1,n)
x2 = np.linspace(0,1,n)
x = np.stack((x1,x2), axis=-1)
# xs, ys = np.meshgrid(x,y,sparse=True)
y = x1 **2 + x2**2
y.resize((n,1))
# print(x.shape, y.shape, z.shape)
likelihoods_list = [Gaussian(sigma=1.5)]
likelihood = HetLikelihood(likelihoods_list)
Y_metadata = likelihood.generate_metadata() # f_index, d_index
ls_q = np.array(([.05] * Q))
var_q = np.array(([.5] * Q))
util.latent_functions_prior(Q, lenghtscale=ls_q, variance=var_q, input_dim=2)
Z = x
print("Z.shape", Z.shape)
kern_list = util.latent_functions_prior(Q, lenghtscale=ls_q, variance=var_q, input_dim=2)
true_W_list = [np.array(([[1.0]]))]

online_model = ContinualGP(X=[x], Y=[y], Z=Z, kern_list=kern_list, 
                           kern_list_old=kern_list, likelihood=likelihood, Y_metadata=Y_metadata)

X_new = np.array([
    [0.1,0.1],
    [0.5, 0.5]
])



online_model = onlineVEM(online_model, vem_iters=VEM_its, optZ=False, verbose=False, verbose_plot=False, non_chained=False)


mu , _ = online_model.predictive_new(X_new)
print "predictive_new"
print mu

print("Xnew.ndim:", X_new.ndim)
mu , _ = online_model._raw_predict_f(X_new)  # predictive still have problems, it includes likelihood (sigma)
print "predictive"
print mu

# mu , _ = online_model._raw_predict_stochastic(X_new)
# print "_raw_predict_stochastic"
# print mu