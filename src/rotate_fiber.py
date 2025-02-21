import numpy as np


def rotate_fiber(center, fiber_length, direction, rot_val, num_points=100):
    # 初期位置を指定する
    half_length = fiber_length / 2
    fiber_start = np.array([center[0] - half_length, center[1] - half_length, center[2] - half_length])
    fiber_end = np.array([center[0] + half_length, center[1] + half_length, center[2] + half_length])

    theta_xy = np.random.normal(0, rot_val, size=num_points)
    all_rotation_vector = []

    for theta in theta_xy:
        theta_xy_rad = np.radians(theta)

        # xy平面での回転行列
        x1 = fiber_start[0] * np.cos(theta_xy_rad) - fiber_start[1] * np.sin(theta_xy_rad)
        y1 = fiber_start[0] * np.sin(theta_xy_rad) + fiber_start[1] * np.cos(theta_xy_rad)
        x2 = fiber_end[0] * np.cos(theta_xy_rad) - fiber_end[1] * np.sin(theta_xy_rad)
        y2 = fiber_end[0] * np.sin(theta_xy_rad) + fiber_end[1] * np.cos(theta_xy_rad)

        # xy平面で回転させたファイバーの始点と終点
        fiber_start_xy = np.array([x1, y1, fiber_start[2]])
        fiber_end_xy = np.array([x2, y2, fiber_end[2]])

        # 円の描画
        def plot_circle(center, fiber_start, fiber_end, fiber_length, num_points=100):
            # 正規化
            normal = fiber_end - fiber_start
            normal = normal / np.linalg.norm(normal)
            
            # 任意のベクトルを作成し、法線ベクトルと直交するベクトルを計算
            if (normal == np.array([0, 0, 1])).all():
                orthogonal_vector = np.array([1, 0, 0])
            else:
                orthogonal_vector = np.cross(normal, np.array([0, 0, 1]))
            
            orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
            
            # もう一つの直交するベクトルを計算
            orthogonal_vector2 = np.cross(normal, orthogonal_vector)
            
            # 円中の点を計算
            theta = np.linspace(0, 2 * np.pi, num_points)
            r = np.sqrt(np.random.uniform(0, 1, num_points)) * fiber_length / 2
            circle_points = center + np.outer(r, orthogonal_vector) * np.cos(theta[:, np.newaxis]) + np.outer(r, orthogonal_vector2) * np.sin(theta[:, np.newaxis])
            
            random_index = np.random.randint(num_points)
            random_point_on_circle = circle_points[random_index]

            # 円周上の点を計算
            circle_points = np.array([center + fiber_length / 2 * (np.cos(t) * orthogonal_vector + np.sin(t) * orthogonal_vector2) for t in theta])

            return random_point_on_circle

        random_point_on_circle = plot_circle(fiber_end_xy, fiber_start_xy, fiber_end, fiber_length)

        direction_vector = random_point_on_circle - center
        direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
        fiber_start_rotated = center - direction_vector_normalized * fiber_length
        rotation_vector = random_point_on_circle - fiber_start_rotated

        # directionを考慮した回転行列を適用
        angle = np.arccos(np.dot(rotation_vector, direction) / (np.linalg.norm(rotation_vector) * np.linalg.norm(direction)))
        axis = np.cross(rotation_vector, direction)
        axis = axis / np.linalg.norm(axis)
        
        K = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        corrected_rotation_vector = np.dot(R, rotation_vector)
        all_rotation_vector.append(corrected_rotation_vector)
        
    return all_rotation_vector
