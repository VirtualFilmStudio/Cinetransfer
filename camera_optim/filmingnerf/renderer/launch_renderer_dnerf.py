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
import pyrender
from pyrender import RenderFlags

from .renderer import Renderer, Transform

from .smpl_model import run_smpl, SmplModel, smpl_to_geometry
from .geometry_np.camera import camera_look_at, spherical_sample
from .geometry.mesh import make_mesh, get_scene_bb

def load_layouts(tracks_path, smpl_model_path, vis_mask, track_ids):
    data = np.load(tracks_path)
    res = {}
    for name in data.files:
        res.update({name: torch.tensor(data[name])})

    B, T, _ = res["trans"].shape

    smpl_model = SmplModel(smpl_model_path, batch_size=B * T)

    with torch.no_grad():
        world_smpl = run_smpl(
            smpl_model,
            res["trans"],
            res["root_orient"],
            res["pose_body"],
            res.get("betas", None),
        )

    smpl_geometries = smpl_to_geometry(
        world_smpl["vertices"], world_smpl["faces"], vis_mask, track_ids
    )
    return smpl_geometries, world_smpl, res["floor_plane"], res["cam_R"][0], res["cam_t"][0]


def spherical_camera_pose(n_samples, center, radius):
    sample_pos = spherical_sample(n_samples)
    up = np.array([0, 1, 0])
    cam_poses = []
    for pos in sample_pos:
        src_pos = [i * radius for i in pos]
        camera_pose = camera_look_at(src_pos, np.array(center), up)
        cam_poses.append(camera_pose)
    return cam_poses


def prep_shared_geometry(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (F, 3)
    :param colors (B, 4)
    """
    B, V, _ = verts.shape
    F, _ = faces.shape
    colors = colors.unsqueeze(1).expand(B, V, -1)[..., :3]
    faces = faces.unsqueeze(0).expand(B, F, -1)
    return verts, faces, colors


def openJson(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def saveJson(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def launch_renderer_dnerf(layouts, smpl_info, cfg, viewer=False):
    save_root = os.path.join(cfg.data_root + "/", cfg.seq_name + "/", "dnerf/")
    os.makedirs(save_root, exist_ok=True)
    img_save_path = os.path.join(save_root, "train")
    os.makedirs(img_save_path, exist_ok=True)
    out_flags = os.listdir(img_save_path)
    mapping_ids = smpl_info["mapping_ids"]
    smpl_joints = smpl_info["joints"]

    if len(out_flags) > 0 and not cfg.overwrite:
        offset_centers = openJson(f"{save_root}/offset_center.json")
        mask_colors = openJson(f"{save_root}/mask_colors.json")
        return save_root, offset_centers, mask_colors

    radius = cfg.render_camera_radius
    verts, colors, faces, bounds = layouts
    T = len(verts)
    time_step = 1 / (T - 1)
    cam_poses = spherical_camera_pose(T, [0, 0, 0], radius)
    h, w = cfg.nerf_train_img_hw
    renderer = Renderer(w, h, alight=(255, 255, 255))
    flags = RenderFlags.RGBA
    renderer.add_camera(cam_poses[0])

    aspect = w / h
    xfov = renderer.camera.yfov * aspect
    trans_json = {
        "camera_angle_x": xfov,
        "camera_angle_y": renderer.camera.yfov,
        "aabb_scale": cfg.aabb_scale,
        "h": h,
        "w": w,
        "frames": []
    }

    frames = []
    offset_centers = []
    mask_colors = []
    for t in range(cfg.start_id, cfg.end_id):
        # load mesh
        meshes = []
        mesh_colors = []
        py_meshes = []

        for k in range(len(verts[t]))[cfg.track_sid : cfg.track_eid]:
            mesh = make_mesh(verts[t][k], faces[t], colors[t][k][:3])
            meshes.append(mesh)
            mesh_colors.append(colors[t][k][:3].tolist())
        bb_min, bb_max = get_scene_bb(meshes)
        center = 0.5 * (bb_min + bb_max)
        center = [-center[0], -center[1], -center[2]]

        offset_centers.append(center)
        mask_colors.append(mesh_colors)

        mesh_trans = Transform((center[0], center[1], center[2]))
        light_trans = Transform((10, 10, 10))
        for k, mesh in enumerate(meshes):
            node = renderer.add_mesh(mesh, mesh_trans)
            py_meshes.append(node)

        cam_node = renderer.camera_node
        renderer.scene.set_pose(cam_node, cam_poses[t])
        rgb, depth = renderer.render(flags)
        H, W, _ = rgb.shape
        rgb_copy = rgb.copy()
        for h in range(H):
            for w in range(W):
                if depth[h][w] == 0:
                    rgb_copy[h][w][3] = 0

        for node in py_meshes:
            node.mesh.is_visible = False

        img = Image.fromarray(rgb_copy)
        img.save(os.path.join(img_save_path, "r_{}.png".format(t)), format="png")
        depth *= 255
        depth = depth.astype(np.uint8)
        imageio.imsave(os.path.join(img_save_path, "r_{}_depth.png".format(t)), depth)

        file_path = f"./train/r_{t}"
        frames.append(
            {
                "file_path": file_path,
                "time": t * time_step,
                "transform_matrix": cam_poses[t].tolist(),
            }
        )
    trans_json["frames"] = frames
    saveJson(f"{save_root}/transforms_train.json", trans_json)
    trans_json["frames"] = frames[::5]
    saveJson(f"{save_root}/transforms_val.json", trans_json)
    trans_json["frames"] = frames[::10]
    saveJson(f"{save_root}/transforms_test.json", trans_json)
    saveJson(f"{save_root}/offset_center.json", offset_centers)
    saveJson(f"{save_root}/mask_colors.json", mask_colors)
    return save_root, offset_centers, mask_colors


def gen_nerf_config_dnerf(save_root, exp_name):
    txt_path = os.path.join(save_root, f"{exp_name}.txt")
    basedir = os.path.join(save_root, "dlogs")
    os.makedirs(basedir, exist_ok=True)
    with open(txt_path, "w") as f:
        f.write(f"expname = {exp_name}\n")
        f.write(f"basedir = {basedir}\n")
        f.write(f"datadir = {save_root}\n")
        f.write(f"dataset_type = blender\n")
        f.write("\n")
        f.write("nerf_type = direct_temporal\n")
        f.write("no_batching = True\n")
        f.write("not_zero_canonical = False\n")
        f.write("use_viewdirs = True\n")
        f.write("\n")
        f.write("use_viewdirs = True\n")
        f.write("white_bkgd = True\n")
        f.write("lrate_decay = 500\n")
        f.write("\n")
        f.write("N_samples = 64\n")
        f.write("N_importance = 128\n")
        f.write("N_rand = 500\n")
        f.write("testskip = 1\n")
        f.write("\n")
        f.write("precrop_iters = 500\n")
        f.write("precrop_iters_time = 100000\n")
        f.write("precrop_frac = 0.5\n")
        f.write("\n")
        f.write("half_res = True\n")
        f.write("do_half_precision = False\n")
    return txt_path
