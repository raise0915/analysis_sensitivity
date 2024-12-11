import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import icecream as ic 

samples = 1000

# Convert Cartesian coordinates (x, y, z) to spherical coordinates (phi, mtheta)
def cartesian_to_spherical(x, y, z):
    mtheta = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))
    phi = np.arctan2(y, x)
    return phi, mtheta

# Example usage
phi, mtheta = cartesian_to_spherical(-10, 50, 24)
ic.ic(phi, mtheta)

# Generate random data for theta following a Gaussian distribution
rng = default_rng()
std_dev = round(15/3, 2)
theta = rng.normal(loc=phi, scale=std_dev, size=samples)  # 3 sigma = 5 degrees
r = 1

# Convert polar coordinates to Cartesian coordinates for 2D plotting
theta = np.radians(theta)

x = r * np.cos(theta)
y = r * np.sin(theta)

# Create a 2D plot
fig, ax = plt.subplots()
ax.scatter(x, y)

# Add arrows pointing towards the center
for i in range(samples):
    ax.arrow(x[i], y[i], -x[i], -y[i], head_width=0.05, head_length=0.1, fc='red', ec='red')

# Set equal scaling
ax.set_aspect('equal', 'box')

plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Rotation with Gaussian Distributed')
plt.show()

# Generate random data for phi following a uniform distribution
phi = rng.uniform(0, 2 * np.pi, samples)

x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

for i in range(samples):
    ax.quiver(0, 0, 0, x[i], y[i], z[i],  length=1, color='red')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Rotation with Gaussian Distributed Theta')

plt.show()