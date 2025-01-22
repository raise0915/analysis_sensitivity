import os
from mayavi import mlab
# os.environ['ETS_TOOLKIT']='qt4'


from cylinder import plot_rotated_cylinder_for_slider # move_second_position
import numpy as np  # noqa: E402

file_path = "089_tumour_model.bin"
# 1. ファイルをバイナリモードで開く
with open(file_path, 'rb') as f:
    data = np.fromfile(f, dtype=np.uint8).reshape([600, 600, 600], order='F')


# x, y, z = plot_rotated_cylinder_for_slider(self.position, self.radius, self.height, self.direction)

mlab.figure()
mlab.contour3d(data, contours=8, opacity=0.5)
# mlab.plot3d(x, y, z, tube_radius=0.5, color=(1, 0, 0))
mlab.show()