import json
import trimesh
import torch
import torch.nn as nn
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

from filmingnerf.renderer.launch_renderer import launch_renderer, load_smpl, load_layouts
from filmingnerf.renderer import make_4x4_pose
from filmingnerf.renderer.geometry.camera import *
from filmingnerf.renderer.geometry.rotation import *
from filmingnerf.camera_init import *

from video_params import check_params

def openJson(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def prepare_data(cfg, dataset, device):
    data = openJson(f"{cfg.out_dir}/{cfg.seq_name}/optim_cams.json")
    data = torch.Tensor(data).to(device)
    
    if 1:
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
        for t in range(len(data)):
            meshes = []
            for k in range(len(verts[t]))[cfg.track_sid:cfg.track_eid]:
                mesh = make_mesh(verts[t][k], faces[t], colors[t][k][:3])
                meshes.append(mesh)
            bb_min, bb_max = get_scene_bb(meshes)
            center = 0.5 * (bb_min + bb_max)
            center = [center[0], center[1], center[2]]
            data[t,:3,3] += torch.Tensor(center)

    trans = data[:, :3, 3]
    rot_m = data[:, :3, :3]
   
    return data, trans, rot_m

class CameraTrjactory(nn.Module):
    def __init__(self, b0, b3):
        super(CameraTrjactory, self).__init__()
        self.b0 = b0
        self.b1 = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.b2 = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.b3 = b3

    def forward(self,t):
        t = torch.tensor(t)
        bt = self.b0 * torch.pow(1-t,3) + self.b1 * 3 * t * torch.pow(1-t, 2) + \
                    self.b2 * 3 * torch.pow(t, 2) * (1-t) + self.b3 * torch.pow(t, 3)
        return bt


def run_opt_position(cfg, dataset, device):
    camera_poses, camera_trans, _ = prepare_data(cfg, dataset, device)
    camTrj = CameraTrjactory(camera_trans[0],camera_trans[-1])
    optimizer = torch.optim.Adam(camTrj.parameters(), weight_decay=0.01)
    step = 1 / (len(camera_trans)-1)
    epochs = 300
    for epoch in range(epochs):
        running_loss = 0.0
        for t in range(len(camera_trans)-2):
            x_ = step * (t+1)
            y_ = camera_trans[t+1]
            
            y_pred = camTrj(x_)
            loss = torch.abs(y_pred - y_).mean()

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch: {epoch + 1:02}/{epochs} Loss: {running_loss:.5e}")

    out_poses = [] 
    renderer = Renderer(150,64,alight=[0.3, 0.3, 0.3],bg_color=[1.0, 1.0, 1.0, 0.0])
    light_trans = Transform((10, 10, 10))
    for t in range(len(camera_poses))[:]:
        renderer.add_light('directlight', light_trans, np.ones(3), 0.9)
        c2w = camera_poses[t].cpu().detach()
        new_trans = camTrj(t*step).cpu().detach()
        c2w[:3, 3] = new_trans
        out_poses.append(c2w.tolist())
        if t%1==0:
            renderer.add_camera(c2w.numpy(), visiable=True)
    # pyrender.Viewer(renderer.scene)
    # import pdb;pdb.set_trace()

    save_root = os.path.join(cfg.out_dir, cfg.seq_name)
    with open(os.path.join(save_root, 'optim_cam_bezier.json'), "w") as f:
        json.dump(out_poses, f, indent=4)


def run_opt_rotation(cfg, device):
    camera_poses, _, rot_matrixes = prepare_data(cfg, device)
    rot_6ds = matrix_to_rotation_6d(rot_matrixes)

    PCA_visualise(rot_6ds.cpu().numpy())
    import pdb;pdb.set_trace()

def PCA_visualise(data):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    projected = pca.fit_transform(data)

    x = projected[:,0]
    y = projected[:,1]
    colors = np.random.rand(len(x))
    plt.scatter(x, y, c=colors, alpha=0.5)
    for i, label in enumerate(range(len(x))):
        plt.annotate(label, (x[i], y[i]))
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.show()
    import pdb;pdb.set_trace()

def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cfg = get_opts()
    cfg = check_params(cfg)
    dataset = get_data_from_cfg(cfg)
    device = get_device(0)
    run_opt_position(cfg, dataset, device)
    # run_opt_rotation(cfg, device)


if __name__ == "__main__":
    main()
