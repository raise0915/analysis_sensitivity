import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import icecream as ic
from mpl_toolkits.mplot3d import Axes3D
rand_uniform01 = np.random.rand


def calculate_position_and_direction(center, radius, length, direction):
    """
    Calculate the new position based on the given center, radius, length, and direction.

    Parameters:
    center (list or tuple): The center coordinates [x, y, z].
    radius (float): The radius of the circle.
    length (float): The length along the z-axis.
    direction (list or tuple): The direction angles [yaw, pitch, roll] in degrees.

    Returns:
    tuple: The new coordinates (x, y, z).
    """
    TWO_PI = 2 * np.pi
    ONE_PI = np.pi

    phi = TWO_PI * rand_uniform01()
    sphi, cphi = np.sin(phi), np.cos(phi)
    r = radius

    ytheta = 2.0 * ONE_PI * direction[0] / 360.0
    ptheta = 2.0 * ONE_PI * direction[1] / 360.0
    rtheta = 2.0 * ONE_PI * direction[2] / 360.0

    Cy, Sy = np.cos(ytheta), np.sin(ytheta)
    Cp, Sp = np.cos(ptheta), np.sin(ptheta)
    Cr, Sr = np.cos(rtheta), np.sin(rtheta)

    x2 = r * cphi
    y2 = r * sphi
    z2 = length / 2.0 - rand_uniform01() * length


    x3 = Cy * Cp * x2 + (Cy * Sp * Sr - Sy * Cr) * y2 + (Cy * Sp * Cr + Sy * Sr) * z2
    y3 = Sy * Cp * x2 + (Sy * Sp * Sr + Cy * Cr) * y2 + (Sy * Sp * Cr - Cy * Sr) * z2
    z3 = -Sp * x2 + Cp * Sr * y2 + Cp * Cr * z2
    
    x3 += center[0]
    y3 += center[1]
    z3 += center[2]

    return x3, y3, z3

def calculate_direction_from_position(x, y, z):
    ONE_PI = np.pi

    r = np.sqrt(x**2 + y**2 + z**2)
    ytheta = np.arctan2(y, x)
    ptheta = np.arccos(z / r)
    rtheta = 0  # Assuming no roll component for simplicity

    direction = [
        ytheta * 360.0 / (2.0 * ONE_PI),
        ptheta * 360.0 / (2.0 * ONE_PI),
        rtheta * 360.0 / (2.0 * ONE_PI)
    ]

    return direction

# Example usage
x3, y3, z3 = calculate_position_and_direction([0, 0, 0], 1, 1, [45, 45, 0])
direction = calculate_direction_from_position(x3, y3, z3)
ic.ic(direction)
    
"""
確認すること
mcxの角度変換がどうなっているか
そこからphiへの変換-> 正規分布 -> 再度直交座標に変換
"""
samples = 5000

# Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, phi, mtheta)
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    mtheta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, phi, mtheta

# Example usage
r, phi, mtheta = cartesian_to_spherical(15, -10, 95)
ic.ic(phi, mtheta)

# Generate random data for theta following a Gaussian distribution
rng = default_rng()
# Convert degrees to radians
std_radian = np.radians(15/3)
ic.ic(std_radian)

theta = rng.normal(loc=phi, scale=std_radian, size=samples)  # 3 sigma = 5 degrees


x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)


# Create a 2D plot
fig, ax = plt.subplots()
ax.scatter(x, y)

# Add arrows pointing towards the center
for i in range(samples):
    ax.arrow(0, 0, x[i], y[i], fc='red', ec='red')

ax.arrow(0, 0, 15, -10, fc='blue', ec='blue')

# Set equal scaling
ax.set_aspect('equal', 'box')

plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Rotation with Gaussian Distributed')
plt.show()

# Generate random data for phi following a uniform distribution
phi = rng.uniform(0, 2 * np.pi, samples)

x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
# ax.set_xlim((-1, 1))
# ax.set_ylim((-1, 1))
# ax.set_zlim((-1, 1)) # type: ignore

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z') # type: ignore
ax.set_title('3D Rotation with Gaussian Distributed Theta')

ax.quiver(0, 0, 0, 15, -10, 95, fc='blue', ec='blue')

plt.show()

for i in range(samples):
    ax.quiver(0, 0, 0, x[i], y[i], z[i], length=1, color='red')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z') # type: ignore
ax.set_title('3D Rotation with Gaussian Distributed Theta')

plt.show()

ic.ic(x[5], y[5], z[5])
ic.ic(x[1257], y[1257], z[1257])
ic.ic(x[2500], y[2500], z[2500])
ic.ic(x[3750], y[3750], z[3750])
ic.ic(x[4999], y[4999], z[4999])
