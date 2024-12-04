import os
os.environ['ETS_TOOLKIT']='qt4'

from traits.api import HasTraits, Range, Instance, on_trait_change  # noqa: E402
from traitsui.api import View, Item, Group, HGroup  # noqa: E402
from tvtk.pyface.scene_editor import SceneEditor # noqa: E402
from mayavi.tools.mlab_scene_model import MlabSceneModel  # noqa: E402
from mayavi.core.ui.mayavi_scene import MayaviScene  # noqa: E402

from cylinder import plot_rotated_cylinder_for_slider # move_second_position
import numpy as np  # noqa: E402

file_path = "089_tumour_model.bin"
# 1. ファイルをバイナリモードで開く
with open(file_path, 'rb') as f:
    data = np.fromfile(f, dtype=np.uint8).reshape([600, 600, 600], order='F')


class Visualization(HasTraits):
    position_x = Range(0, 600, 600)
    position_y = Range(0, 600, 600)
    position_z = Range(0, 600, 600)
    direction_x = Range(-180, 180, 360)
    direction_y = Range(-180, 180, 360)
    direction_z = Range(-180, 180, 360)
    """
    position_x1 = Range(-180, 180, 360)
    position_y1 = Range(-180, 180, 360)
    position_z1 = Range(-180, 180, 360)
    direction_x1 = Range(-180, 180, 360)
    direction_y1 = Range(-180, 180, 360)
    direction_z1 = Range(-180, 180, 360)
    """
    scene = Instance(MlabSceneModel, ())
    
    shape = (600, 600, 600)  # Shape of the grid
    position = np.array([289.8410,189.1007,167.2065])  # Center of the cylinder
    direction = np.array([300, 300, 300])  
    radius = 1  # Radius of the cylinder
    height = 22  # Height of the cylinder
    distance = 3

    def __init__(self):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        self.scene.mlab.contour3d(data,contours=5, opacity=0.2)
        self.position = np.array([200, 200, 200])
        self.direction = np.array([0.0, 0, -1])
        self.position1 = np.array([200, 200, 200])
        self.direction1 = np.array([0, 0, -1])
        # x, y, z = plot_cylinder(self.position, self.direction, self.radius, self.height)
        x, y, z = plot_rotated_cylinder_for_slider(self.position, self.radius, self.height, self.direction)
        # potition1 = move_second_position(self.position, self.position1, self.distance)
        # x1, y1, z1 = plot_rotated_cylinder_for_slider(potition1, self.radius, self.height, self.direction1)
        self.plot = self.scene.mlab.points3d(x, y, z, mode='cube', color=(0, 0, 0.8), scale_factor=1, opacity=0.4)
        # self.plot1  = self.scene.mlab.points3d(x1, y1, z1, mode='cube', color=(0.8, 0, 0), scale_factor=1, opacity=0.4)


    @on_trait_change('position_x, position_y, position_z, direction_x, direction_y, direction_z')
    # , position_x1, position_y1, position_z1, direction_x1, direction_y1, direction_z1
    def update_plot(self):
        position = np.array([self.position_x, self.position_y, self.position_z])
        direction = np.array([self.direction_x, self.direction_y, self.direction_z])
        # position1 = np.array([self.position_x1, self.position_y1, self.position_z1])
        # direction1 = np.array([self.direction_x1, self.direction_y1, self.direction_z1])
        x, y, z = plot_rotated_cylinder_for_slider(position, self.radius, self.height, direction)
        # pos1 = move_second_position(position, position1, self.distance)
        # x1, y1, z1 = plot_rotated_cylinder_for_slider(pos1, self.radius, self.height, direction1)
        print('--------------------------------')
        print('position %s' % position)
        # print('position1 %s' % pos1)
        # print('direction %s' % direction)
        # rint('direction1 %s' % direction1)
        print('--------------------------------')
        self.plot.mlab_source.trait_set(x=x, y=y, z=z)
        # self.plot1.mlab_source.trait_set(x=x1, y=y1, z=z1)
               
    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),show_label=False), 
                Group('_', 
                       HGroup('position_x', 'position_y', 'position_z', label='Position'),
                       HGroup('direction_x', 'direction_y', 'direction_z', label='Direction'),
                       # HGroup('position_x1', 'position_y1', 'position_z1', label='Position1'),
                       # HGroup('direction_x1', 'direction_y1', 'direction_z1', label='Direction1'),
                       ),
                resizable=True,
                )


visualization = Visualization()
visualization.configure_traits()
