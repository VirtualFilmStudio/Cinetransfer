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

from filmingnerf.renderer.launch_renderer import launch_renderer, load_smpl
from filmingnerf.renderer import make_4x4_pose
from filmingnerf.renderer.geometry.camera import *
from filmingnerf.renderer.geometry.rotation import *
from filmingnerf.camera_init import *

from video_params import check_params

def run_smpl2bvh(cfg, dataset, device):
    B = len(dataset)
    T = dataset.seq_len

    loader = DataLoader(dataset, batch_size=int(T), shuffle=False)

    obs_data = move_to(next(iter(loader)), device)

    vis_mask = dataset.get_vis_mask()
    track_ids = dataset.get_track_id()

    tracks_path = os.path.join(cfg.data_root, cfg.seq_name, f"{cfg.seq_name}.npz")
    smpl_model_path = os.path.join(cfg.data_root, 'smpl_tmpl_model.npz')
    smpl_orients, smpl_translate = load_smpl(tracks_path, smpl_model_path, vis_mask, track_ids)
    
    import pickle
    for i in range(len(smpl_orients)):
        smpl_poses = smpl_orients[i].cpu().numpy()
        smpl_trans = smpl_translate[i].cpu().numpy()
        out = {
            'smpl_poses':smpl_poses,
            'smpl_trans':smpl_trans,
            'smpl_scaling':1
        }
        with open(f'{cfg.seq_name}_{i}.pkl', 'wb') as file:
            pickle.dump(out, file)


def main():
    cfg = get_opts()
    cfg = check_params(cfg)
    dataset = get_data_from_cfg(cfg)
    device = get_device(0)
    
    run_smpl2bvh(cfg, dataset, device)


if __name__ == "__main__":
    main()
