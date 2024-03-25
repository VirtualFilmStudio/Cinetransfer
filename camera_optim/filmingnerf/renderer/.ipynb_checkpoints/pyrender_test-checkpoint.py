import os
import torch
import numpy as np
import trimesh
import imageio
import matplotlib.pyplot as plt

from renderer import Renderer, Transform
from geometry.camera import make_4x4_pose, lookat_matrix
from geometry.mesh import get_mesh_bb

from geometry_np.camera import camera_look_at

mesh_path = '../examples/models/fuze.obj'
save_root = './'

fuze_trimesh = trimesh.load(mesh_path)
renderer = Renderer(400, 400)
s = np.sqrt(2)/2

mesh_trans = Transform((0,0,0))
light_trans = Transform((10,10,10))

source_pos = np.array([1,2,2])
target_pos = np.array([0,0,0])
up = np.array([0,0,1])
camera_pose = camera_look_at(source_pos, target_pos, up)
print(camera_pose)
renderer.add_mesh(fuze_trimesh, mesh_trans)
renderer.add_camera(camera_pose, visiable=True)
renderer.add_light('directlight', light_trans, np.ones(3), 100)

import pyrender
pyrender.Viewer(renderer.scene)
# rgb, depth = renderer.render()
# imageio.imsave(os.path.join(save_root, "rgb.png"), rgb)
# imageio.imsave(os.path.join(save_root, "depth.png"), depth)