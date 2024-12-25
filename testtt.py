import numpy as np
from run_mcx import Runmcx
import pandas as pd
from calc_dvh import calc_log_dvh_with_log
from scipy.interpolate import interp1d
from cylinder import create_rotated_cylinder_mask
import icecream as isc
from numpy.random import default_rng
from matplotlib import pyplot as plt


std_dev = 10/3

samples = []
covariance_matrix = np.diag(
    [std_dev**2, std_dev**2, std_dev**2]
)  # 共分散行列（対角行列）
variables = np.array([1, 2, 3])  # Define the variables array
num_samples = 100  # Define the number of samples
rng = default_rng()
samples = rng.multivariate_normal(variables, covariance_matrix, num_samples)
