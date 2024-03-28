import json
import trimesh
import torch
import pyrender
import numpy as np
from filmingnerf.renderer.geometry.mesh import make_mesh, get_scene_bb
from filmingnerf.renderer.renderer import Renderer, Transform
import os
import imageio
import torch
from einops import rearrange
from tqdm import tqdm
import torchvision.transforms.functional as fn
import torchvision.transforms
from torch.utils.data import DataLoader

from filmingnerf.data import get_data_from_cfg
from filmingnerf.tools.tensor import get_device, move_to
from filmingnerf.opt import get_opts

from filmingnerf.renderer.launch_renderer import launch_renderer, load_layouts
from filmingnerf.renderer import make_4x4_pose
from filmingnerf.renderer.geometry.camera import *
from filmingnerf.camera_init import *

from video_params import check_params

def openJson(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def make_checkerboard(
    length=25.0,
    color0=[0.8, 0.9, 0.9],
    color1=[0.6, 0.7, 0.7],
    tile_width=0.5,
    alpha=1.0,
    up="y",
):
    v, f, _, fc = checkerboard_geometry(length, color0, color1, tile_width, alpha, up)
    return trimesh.Trimesh(v, f, face_colors=fc, process=False)

def fit_plane(points):
    """
    :param points (*, N, 3)
    returns (*, 3) plane parameters (returns in normal * offset format)
    """
    *dims, N, D = points.shape
    mean = points.mean(dim=-2, keepdim=True)
    # (*, N, D), (*, D), (*, D, D)
    U, S, Vh = torch.linalg.svd(points - mean)
    normal = Vh[..., -1, :]  # (*, D)
    offset = torch.einsum("...ij,...j->...i", points, normal)  # (*, N)
    offset = offset.mean(dim=-1, keepdim=True)
    return torch.cat([normal, offset], dim=-1)


def get_plane_transform(up, ground_plane=None, xyz_orig=None):
    """
    get R, t rigid transform from plane and desired origin
    :param up (3,) up vector of coordinate frame
    :param ground_plane (4) (a, b, c, d) where a,b,c is the normal
    :param xyz_orig (3) desired origin
    """
    R = torch.eye(3)
    t = torch.zeros(3)
    if ground_plane is None:
        return R, t
    # compute transform between world up vector and passed in floor
    ground_plane = torch.as_tensor(ground_plane)
    ground_plane = torch.sign(ground_plane[3]) * ground_plane

    normal = ground_plane[:3]
    normal = normal / torch.linalg.norm(normal)
    v = torch.linalg.cross(up, normal)
    ang_sin = torch.linalg.norm(v)
    ang_cos = up.dot(normal)
    skew_v = torch.as_tensor([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    R = torch.eye(3) + skew_v + (skew_v @ skew_v) * ((1.0 - ang_cos) / (ang_sin**2))

    # project origin onto plane
    if xyz_orig is None:
        xyz_orig = torch.zeros(3)
    t, _ = compute_plane_intersection(xyz_orig, -normal, ground_plane)

    return R, t


def parse_floor_plane(floor_plane):
    """
    Takes floor plane in the optimization form (Bx3 with a,b,c * d) and parses into
    (a,b,c,d) from with (a,b,c) normal facing "up in the camera frame and d the offset.
    """
    floor_offset = torch.norm(floor_plane, dim=-1, keepdim=True)
    floor_normal = floor_plane / floor_offset

    # in camera system -y is up, so floor plane normal y component should never be positive
    #       (assuming the camera is not sideways or upside down)
    neg_mask = floor_normal[..., 1:2] > 0.0
    floor_normal = torch.where(
        neg_mask.expand_as(floor_normal), -floor_normal, floor_normal
    )
    floor_offset = torch.where(neg_mask, -floor_offset, floor_offset)
    floor_plane_4d = torch.cat([floor_normal, floor_offset], dim=-1)

    return floor_plane_4d


def compute_plane_intersection(point, direction, plane):
    """
    Given a ray defined by a point in space and a direction,
    compute the intersection point with the given plane.
    Detect intersection in either direction or -direction.
    Note, ray may not actually intersect with the plane.

    Returns the intersection point and s where
    point + s * direction = intersection_point. if s < 0 it means
    -direction intersects.

    - point : B x 3
    - direction : B x 3
    - plane : B x 4 (a, b, c, d) where (a, b, c) is the normal and (d) the offset.
    """
    dims = point.shape[:-1]
    plane_normal = plane[..., :3]
    plane_off = plane[..., 3]
    s = (plane_off - bdot(plane_normal, point)) / (bdot(plane_normal, direction) + 1e-4)
    itsct_pt = point + s.reshape((-1, 1)) * direction
    return itsct_pt, s


def bdot(A1, A2, keepdim=False):
    """
    Batched dot product.
    - A1 : B x D
    - A2 : B x D.
    Returns B.
    """
    return (A1 * A2).sum(dim=-1, keepdim=keepdim)

def checkerboard_geometry(
    length=12.0,
    color0=[0.8, 0.9, 0.9],
    color1=[0.6, 0.7, 0.7],
    tile_width=0.5,
    alpha=1.0,
    up="y",
):
    assert up == "y" or up == "z"
    color0 = np.array(color0 + [alpha])
    color1 = np.array(color1 + [alpha])
    radius = length / 2.0
    num_rows = num_cols = int(length / tile_width)
    vertices = []
    vert_colors = []
    faces = []
    face_colors = []
    for i in range(num_rows):
        for j in range(num_cols):
            u0, v0 = j * tile_width - radius, i * tile_width - radius
            us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
            vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
            zs = np.zeros(4)
            if up == "y":
                cur_verts = np.stack([us, zs, vs], axis=-1)  # (4, 3)
            else:
                cur_verts = np.stack([us, vs, zs], axis=-1)  # (4, 3)

            cur_faces = np.array(
                [[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64
            )
            cur_faces += 4 * (i * num_cols + j)  # the number of previously added verts
            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            cur_color = color0 if use_color0 else color1
            cur_colors = np.array([cur_color, cur_color, cur_color, cur_color])

            vertices.append(cur_verts)
            faces.append(cur_faces)
            vert_colors.append(cur_colors)
            face_colors.append(cur_colors)

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    vert_colors = np.concatenate(vert_colors, axis=0).astype(np.float32)
    faces = np.concatenate(faces, axis=0).astype(np.float32)
    face_colors = np.concatenate(face_colors, axis=0).astype(np.float32)

    return vertices, faces, vert_colors, face_colors

def camera_marker_geometry(radius, height, up):
    assert up == "y" or up == "z"
    if up == "y":
        vertices = np.array(
            [
                [-radius, -radius, 0],
                [radius, -radius, 0],
                [radius, radius, 0],
                [-radius, radius, 0],
                [0, 0, height],
            ]
        )
    else:
        vertices = np.array(
            [
                [-radius, 0, -radius],
                [radius, 0, -radius],
                [radius, 0, radius],
                [-radius, 0, radius],
                [0, -height, 0],
            ]
        )

    faces = np.array(
        [[0, 3, 1], [1, 3, 2], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],]
    )

    face_colors = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    return vertices, faces, face_colors

def transform_pyrender(T_c2w):
    """
    :param T_c2w (*, 4, 4)
    """
    T_vis = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=T_c2w.device,
    )
    return torch.einsum(
        "...ij,jk->...ik", torch.einsum("ij,...jk->...ik", T_vis, T_c2w), T_vis
    ).cpu().numpy()

def transform_slam(T_c2w):
    """
    :param T_c2w (*, 4, 4)
    """
    T_vis = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=T_c2w.device,
    )
    T_vis = torch.linalg.inv(T_vis)
    return torch.einsum(
        "...ij,jk->...ik", torch.einsum("ij,...jk->...ik", T_vis, T_c2w), T_vis
    ).cpu().numpy()


def run_vis(cfg, dataset, device):
    B = len(dataset)
    T = dataset.seq_len

    loader = DataLoader(dataset, batch_size=int(T), shuffle=False)

    obs_data = move_to(next(iter(loader)), device)

    vis_mask = dataset.get_vis_mask()
    track_ids = dataset.get_track_id()

    tracks_path = os.path.join(cfg.data_root, cfg.seq_name, f"{cfg.seq_name}.npz")
    smpl_model_path = os.path.join(cfg.data_root, 'smpl_tmpl_model.npz')
    layouts, world_smpl, floor_plane, cam_R, cam_t = load_layouts(tracks_path, smpl_model_path, vis_mask, track_ids)
    verts, colors, faces, bounds = layouts

    meshes = []
    use_opti = 1
    use_slam = not use_opti
    smooth = 1
    save_camera = 0
    if use_opti:
        data = openJson(f"{cfg.out_dir}/{cfg.seq_name}/optim_cams.json")
        data = np.asarray(data)
        if smooth:
            data = trans_matrix_smooth(data,n=30)

            if save_camera:
                data = transform_slam(torch.Tensor(data))

                save_root= os.path.join(cfg.data_root+'/', cfg.seq_name+'/', 'dnerf/')
                with open(os.path.join(save_root, 'transforms_test.json'), 'r') as f:
                    meta = json.load(f)
                xfov = float(meta['camera_angle_x'])
                yfov = float(meta['camera_angle_y'])
                h = 1080
                w = 2534
                focal_w = 0.5*w/np.tan(0.5*xfov)
                focal_y = 0.5*h/np.tan(0.5*yfov)
            
                frame_w2c = torch.linalg.inv(torch.Tensor(data)).cpu().numpy()
                intrins = np.array([[focal_w, focal_y, w / 2, h / 2]])
                intrins = np.repeat(intrins, len(data), axis=0)
                
                np.savez(
                    f"{cfg.out_dir}/{cfg.seq_name}/cameras.npz",
                    height=h,
                    width=w,
                    focal=0,
                    intrins=intrins[:, :4],
                    w2c=frame_w2c,
                )
                return 0
            
    elif use_slam:
        camera_poses_w2c = make_4x4_pose(cam_R, cam_t)
        camera_poses_c2w = torch.linalg.inv(camera_poses_w2c)
        camera_poses_c2w = transform_pyrender(camera_poses_w2c)
        camera_poses_c2w = transform_slam(torch.Tensor(camera_poses_c2w))
        data = camera_poses_c2w
        data[:,3,:] = np.array([0,0,0,1])
        
    H,W = cfg.nerf_render_img_hw
    renderer = Renderer(cfg.downsample*W,cfg.downsample*H,alight=[0.3, 0.3, 0.3],bg_color=[1.0, 1.0, 1.0, 0.0])
    light_trans = Transform((10, 10, 10))
    
    renderer.add_camera(data[0], visiable=True)
    ground_pose = np.eye(4)
    ground = pyrender.Mesh.from_trimesh(
        make_checkerboard(up="y", alpha=1.0), smooth=False
    )
    tid, sid = torch.where(vis_mask > 0)
    idx = tid[torch.argmin(sid)]
    root = world_smpl["joints"][idx, 0, 0].detach().cpu()
    floor = parse_floor_plane(floor_plane.detach().cpu())
    R, t = get_plane_transform(torch.tensor([0.0, 1.0, 0.0]).cpu(), floor[0], root)
    ground_pose = make_4x4_pose(R.cpu(), t).numpy()
    ground_pose = transform_pyrender(torch.Tensor(ground_pose))
   
    
    imgs = []
    for t in range(len(data))[:]:
        ground_node = renderer.scene.add(ground, name="ground", pose=ground_pose)
        renderer.add_light('directlight', light_trans, np.ones(3), 0.9)
        c2w = data[t]
        renderer.add_camera(c2w, visiable=True)
        for k in range(len(verts[t]))[cfg.track_sid:cfg.track_eid]:
            mesh = make_mesh(verts[t][k], faces[t], colors[t][k][:3])
            meshes.append(mesh)
        bb_min, bb_max = get_scene_bb(meshes)
        center = 0.5 * (bb_min + bb_max)
        center = [center[0], center[1], center[2]]
        if use_opti:
            c2w[:3,3] = c2w[:3, 3] + center
        mesh_trans = Transform((-center[0], -center[1], -center[2]))

        meshes = []
        for k in range(len(verts[t])):
            mesh = make_mesh(verts[t][k], faces[t], colors[t][k][:3])
            meshes.append(mesh)
        for mesh in meshes:
            renderer.scene.add_node(
            pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh)))

        cam_node = renderer.camera_node
        renderer.scene.set_pose(cam_node, c2w)
        rgb, depth = renderer.render(pyrender.RenderFlags.NONE)
        imgs.append(rgb)
        renderer.scene.clear()
        meshes = []


    img_save_path = os.path.join(cfg.out_dir, cfg.seq_name)

    if use_opti:
        if smooth:
            imageio.mimwrite(os.path.join(img_save_path, '3d_view_smooth.gif'), imgs, fps=30)
        else:
            imageio.mimwrite(os.path.join(img_save_path, '3d_view.gif'), imgs, fps=30)
    else:
        imageio.mimwrite(os.path.join(img_save_path, '3d_view_slam.gif'), imgs, fps=30)

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

from filmingnerf.renderer.geometry.rotation import *
def trans_matrix_smooth(data, n=10):
    rotations = data[:,:3,:3]
    rotations = matrix_to_rotation_6d(torch.Tensor(rotations)).cpu().numpy()
    positions = data[:,:3,3]
    
    positions = moving_average(positions, n)
    rotations = moving_average(rotations, n)
    rotations = rotation_6d_to_matrix(torch.Tensor(rotations)).cpu().numpy()
    
    data[n-1:, :3, :3] = rotations
    data[n-1:, :3, 3] = positions
    return data

def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cfg = get_opts()
    cfg = check_params(cfg)
    dataset = get_data_from_cfg(cfg)
    device = torch.device('cpu')
    run_vis(cfg, dataset, device)

if __name__ == "__main__":
    main()

