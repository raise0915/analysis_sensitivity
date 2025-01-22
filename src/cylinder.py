import numpy as np
from scipy.spatial.transform import Rotation as R


def create_rotated_cylinder_mask(dims, center, radius, length, direction):    
    full_mask = np.zeros(dims, dtype=bool)
    for _ in range(1000):
        cc = calculate_position_and_direction(np.array(center), radius, length, direction)
        full_mask[int(cc[0]), int(cc[1]), int(cc[2])] = 1
    return full_mask

def plot_rotated_cylinder_for_slider(center, radius, length, direction):   
    x, y, z = [], [], []
    for _ in range(100000):
        cc = calculate_position_and_direction(np.array(center), radius, length, direction)
        x.append(cc[0])
        y.append(cc[1])
        z.append(cc[2])
    return x, y, z

def move_second_position(position, direction, distance):
    position = np.array(position)
    direction = np.array(direction).astype(np.float64)

    vector_length = np.linalg.norm(direction)
    direction /= vector_length
    distance *= 2 # due to 0.5 mm unit
    return position - distance * direction

def rand_uniform01():
    return np.random.rand()

def calculate_position_and_direction(center, radius, length, direction):
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

    x1 = center[0] + r * cphi
    y1 = center[1] + r * sphi
    z1 = center[2] + length / 2.0 - rand_uniform01() * length

    x2, y2, z2 = x1 - center[0], y1 - center[1], z1 - center[2]

    x3 = Cy * Cp * x2 + (Cy * Sp * Sr - Sy * Cr) * y2 + (Cy * Sp * Cr + Sy * Sr) * z2
    y3 = Sy * Cp * x2 + (Sy * Sp * Sr + Cy * Cr) * y2 + (Sy * Sp * Cr - Cy * Sr) * z2
    z3 = -Sp * x2 + Cp * Sr * y2 + Cp * Cr * z2

    center[0] = center[0] + x3
    center[1] = center[1] + y3
    center[2] = center[2] + z3

    return center

# 回転ベクトルを使用したシリンダー回転
def plot_cylinder(center, direction, radius, height):
    direction = np.deg2rad(direction)
    # Create points along the cylinder axis
    z = np.linspace(0, height, 100)

    # Create points in the radial direction
    theta = np.linspace(0, 2 * np.pi, 100)
    theta, z = np.meshgrid(theta, z)

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = center[2] + z
    

    # Calculate rotation angles and axes
    rotation_axis_x = np.cross([1, 0, 0], direction)
    rotation_angle_x = np.arccos(np.dot([1, 0, 0], direction) / np.linalg.norm(direction))
    rotation_axis_y = np.cross([0, 1, 0], direction)
    rotation_angle_y = np.arccos(np.dot([0, 1, 0], direction) / np.linalg.norm(direction))
    rotation_axis_z = np.cross([0, 0, 1], direction)
    rotation_angle_z = np.arccos(np.dot([0, 0, 1], direction) / np.linalg.norm(direction))
    
    # Rotate the cylinder to align with the given direction
    rot_x = R.from_rotvec(rotation_axis_x * rotation_angle_x)
    rot_y = R.from_rotvec(rotation_axis_y * rotation_angle_y)
    rot_z = R.from_rotvec(rotation_axis_z * rotation_angle_z)
    rotation = rot_x * rot_y * rot_z
    
    array = np.array([x.flatten() - center[0], y.flatten() - center[1], z.flatten() - center[2]])
    rotated_points = rotation.apply(array.T).T
    
    x = rotated_points[0].reshape((100, 100)) + center[0]
    y = rotated_points[1].reshape((100, 100)) + center[1]
    z = rotated_points[2].reshape((100, 100)) + center[2]

    return x, y, z


def rotate_around_axis(array, initial_rot, axis, center):
    rotation = R.from_rotvec(initial_rot * np.array(axis))  # 指定された軸周りの回転行列を作成
    rotated_array = rotation.apply(array.T).T
    return rotated_array

