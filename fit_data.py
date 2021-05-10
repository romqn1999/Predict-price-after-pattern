import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel as C


def polyfit(x_data, y_data, degree=80):
    # polynomial fit of degree xx
    pol = np.polyfit(x_data, y_data, degree)
    y_pol = np.polyval(pol, x_data)
    return y_pol


def gaussfit(x_data, y_data, max_len_per_fit=200):
    # Define kernel parameters. 
    l = 0.1
    sigma_f = 2
    # Error standard deviation. 
    sigma_n = 0.3

    # Define kernel object. 
    kernel = C(constant_value=sigma_f,constant_value_bounds=(1e-3, 1e3)) \
                * RBF(length_scale=l, length_scale_bounds=(1e-3, 1e3))

    # Define GaussianProcessRegressor object. 
    gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n**2, n_restarts_optimizer=2)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(np.array(x_data).reshape(-1, 1), y_data)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pol, sigma = gp.predict(np.array(x_data).reshape(-1, 1), return_std=True)
    return y_pol
