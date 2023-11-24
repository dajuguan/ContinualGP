import GPy
import numpy as np
import matplotlib.pyplot as plt


def toy_ARD(
    max_iters=1000, kernel_type="linear", num_samples=300, D=4, optimize=True, plot=True
):
    # Create an artificial dataset where the values in the targets (Y)
    # only depend in dimensions 1 and 3 of the inputs (X). Run ARD to
    # see if this dependency can be recovered
    X1 = np.sin(np.sort(np.random.rand(num_samples, 1) * 10, 0))
    X2 = np.cos(np.sort(np.random.rand(num_samples, 1) * 10, 0))
    X3 = np.exp(np.sort(np.random.rand(num_samples, 1), 0))
    X4 = np.log(np.sort(np.random.rand(num_samples, 1), 0))
    X = np.hstack((X1, X2, X3, X4))

    Y1 = np.asarray(2 * X[:, 0] + 3).reshape(-1, 1)
    Y2 = np.asarray(4 * (X[:, 2] - 1.5 * X[:, 0])).reshape(-1, 1)
    Y = np.hstack((Y1, Y2))

    Y = np.dot(Y, np.random.rand(2, D))
    Y = Y + 0.2 * np.random.randn(Y.shape[0], Y.shape[1])
    Y -= Y.mean()
    Y /= Y.std()

    if kernel_type == "linear":
        kernel = GPy.kern.Linear(X.shape[1], ARD=1)
    elif kernel_type == "rbf_inv":
        kernel = GPy.kern.RBF_inv(X.shape[1], ARD=1)
    else:
        kernel = GPy.kern.RBF(X.shape[1], ARD=1)
    kernel += GPy.kern.White(X.shape[1]) + GPy.kern.Bias(X.shape[1])
    m = GPy.models.GPRegression(X, Y, kernel)
    # len_prior = GPy.priors.inverse_gamma(1,18) # 1, 25
    # m.set_prior('.*lengthscale',len_prior)

    if optimize:
        m.optimize(optimizer="scg", max_iters=max_iters)

    # if plot:
    #     m.kern.plot_ARD()

    return m

# model = toy_ARD(400, "linear", 200)
# X_new = np.array([
#     [0,0,0,0],
#     [1,1,1,1]
# ])
# y = model.predict(X_new,full_cov=True)
# print("Y:", y)


def sparse_GP_regression_2D(
    num_samples=400, num_inducing=50, max_iters=100, optimize=True, plot=True, nan=False
):
    """Run a 2D example of a sparse GP regression."""
    np.random.seed(1234)
    X = np.random.uniform(-3.0, 3.0, (num_samples, 2))
    # Y = np.sin(X[:, 0:1]) * np.sin(X[:, 1:2]) + np.random.randn(num_samples, 1) * 0.05
    Y1 = np.asarray(2 * X[:, 0] + 3).reshape(-1, 1)
    Y2 = np.asarray(4 * (X[:, 1] - 1.5 * X[:, 0])).reshape(-1, 1)
    Y = np.hstack((Y1, Y2))
    if nan:
        inan = np.random.binomial(1, 0.2, size=Y.shape)
        Y[inan] = np.nan

    # construct kernel
    rbf = GPy.kern.RBF(2)
    print("rbf:", rbf)

    # create simple GP Model
    m = GPy.models.SparseGPRegression(X, Y, kernel=rbf, num_inducing=num_inducing)

    # contrain all parameters to be positive (but not inducing inputs)
    m[".*len"] = 2.0

    m.checkgrad()

    # optimize
    if optimize:
        m.optimize("tnc", messages=1, max_iters=max_iters)

    # plot
    # if MPL_AVAILABLE and plot:
    #     m.plot()

    print(m)
    return m

model = sparse_GP_regression_2D(num_samples=200, num_inducing=50, max_iters=100)
X_new = np.array([
    [np.pi/2,np.pi/2],
    [0,0]
])
y = model.predict(X_new,full_cov=True)
print("Y:", y)