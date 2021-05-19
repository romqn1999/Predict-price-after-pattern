import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel as C


def polyfit(x_data, y_data, degree=80):
    # polynomial fit of degree xx
    pol = np.polyfit(x_data, y_data, degree)
    y_pol = np.polyval(pol, x_data)
    return y_pol


def gaussfit(x_data, y_data, chunk_size=200):
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

    y_res = np.empty((0))
    # Split data into sized chunks
    for i in range(0, len(x_data), chunk_size):
        x_chunk = x_data[i: i + chunk_size]
        y_chunk = y_data[i: i + chunk_size]
        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(np.array(x_chunk).reshape(-1, 1), y_chunk)
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pre, sigma = gp.predict(np.array(x_chunk).reshape(-1, 1), return_std=True)
        y_res.append(y_pre)
    return y_res
