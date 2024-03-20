import numpy as np
import torch
import os
import torch
import json
import numpy as np
import trimesh
import imageio
from PIL import Image
import matplotlib.pyplot as plt

from .renderer import Renderer, Transform
from .geometry.camera import make_4x4_pose, lookat_matrix

from .smpl_model import run_smpl, SmplModel
from .geometry_np.camera import camera_look_at, spherical_sample
from .geometry_np.mesh import make_mesh, get_scene_bb
 
# path = '../../data/smpl/arc1_000360_world_results.npz'
def test():
    path = '../data/smpl/follow0.npz'
    bm_path = '../data/smpl_tmpl_model.npz'

    data = np.load(path)
    res = {}
    for name in data.files:
        res.update({name: torch.tensor(data[name])})

    B, T, _ = res["trans"].shape
    smpl_model = SmplModel(bm_path, batch_size=B*T)

    with torch.no_grad():
        world_smpl = run_smpl(
            smpl_model, 
            res["trans"],
            res["root_orient"],
            res["pose_body"],
            res.get("betas", None)
        )
    # verts: (B, T, V, 3) -> (human num, time, vertices)
    # faces: (F, 3) 
    verts = world_smpl['vertices'].numpy()
    faces = world_smpl['faces'].numpy()

    # verts: T list (B, V, 3)
    # faces: T list (F, 3)
    B, T, V = verts.shape[:3]
    faces = [faces for i in range(T)]
    verts = [verts[:, t] for t in range(T)]
    colors = [[255,0,0] for i in range(V)]
    print(B,T)

    t = 10
    id = 0
    meshes = []
    for k in range(B):
        mesh = make_mesh(verts[t][k], faces[t], colors)
        meshes.append(mesh)

    save_root = '../output/smpl'
    renderer = Renderer(400, 400)
    s = np.sqrt(2)/2

    meshes = meshes[:2]

    bb_min, bb_max = get_scene_bb(meshes)
    center = 0.5 * (bb_min + bb_max)

    radius  = 3
    n_samples = 100
    sample_pos = spherical_sample(n_samples)
    up = np.array([0,1,0])
    cam_poses = []
    for pos in sample_pos:
        # import pdb;pdb.set_trace()
        src_pos = [i*radius for i in pos]
        camera_pose = camera_look_at(src_pos, np.array([0,0,0]), up)
        cam_poses.append(camera_pose)

    mesh_trans = Transform((-center[0], -center[1], -center[2]))
    light_trans = Transform((10, 10, 10))
    for mesh in meshes:
        renderer.add_mesh(mesh, mesh_trans)

    # for cam_pose in cam_poses: 
    #     renderer.add_camera(cam_pose, visiable=True)
    renderer.add_camera(cam_poses[0])

    renderer.add_light('directlight', light_trans, np.ones(3), 10)

    from pyrender import RenderFlags
    import pyrender

    flags = RenderFlags.RGBA
    # flags = RenderFlags.NONE
    # pyrender.Viewer(renderer.scene)
    for i in range(n_samples):
        cam_node = renderer.camera_node
        renderer.scene.set_pose(cam_node, cam_poses[i])
        rgb, depth = renderer.render(flags)
        H, W, _ = rgb.shape
        rgb_copy = rgb.copy()
        for h in range(H):
            for w in range(W):
                if depth[h][w] == 0:
                    rgb_copy[h][w][3] = 0
        img = Image.fromarray(rgb_copy)
        img.save(os.path.join(save_root, 'images', "r_{}.png".format(i)), format='png')
        imageio.imsave(os.path.join(save_root, 'images', "r_{}_depth_{:0>4d}.png".format(i, i)), depth)

    def saveJson(filename, data):
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    trans_json = {
                    "camera_angle_x": renderer.camera.yfov,
                    "aabb_scale": 1,
                    "frames":[]
                }

    for i in range(n_samples):
        file_path = f'./images/r_{i}'
        trans_json['frames'].append(
            {
                "file_path": file_path,
                "transform_matrix":cam_poses[i].tolist()
            }
        )

    saveJson(f'{save_root}/transforms.json', trans_json)