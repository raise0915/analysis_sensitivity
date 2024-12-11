import numpy as np
from numpy.random import default_rng
import icecream as ic

num_samples = 100
std_dev = {"position": 5, "direction": 5}
variables = [[0, 0, 0]]

std_dev_pos = std_dev["position"]
std_dev_dir = std_dev["direction"]

# position
covariance_matrix = np.diag([std_dev_pos**2, std_dev_pos**2, std_dev_pos**2])  # 共分散行列（対角行列）
variables = np.concatenate(variables)
rng = default_rng()
samples_pos = rng.multivariate_normal(variables, covariance_matrix, num_samples)

# direction
rng = default_rng()
theta = rng.normal(loc=0, scale=std_dev_dir, size=num_samples)  # 3 sigma = 5 degrees
phi = rng.uniform(0, 2 * np.pi, num_samples)
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

samples = {
    "position": samples_pos,
    "direction": np.array([x, y, z]).T
}


ic.ic(samples['direction'])


for i in range(1):
    for key in samples:
        if key == "position":
            cylinder_pos = samples[key][i]
        elif key == "direction":
            cylinder_rot = samples[key][i]
    ic.ic(cylinder_pos)
    ic.ic(samples["position"][0])
    ic.ic(cylinder_rot)
