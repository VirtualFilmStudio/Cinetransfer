import cv2
import os
import time
import json
import imageio
import numpy as np

import torch
from einops import rearrange
from tqdm import tqdm
import torchvision.transforms.functional as fn
import torchvision.transforms
from torch.utils.data import DataLoader
# torch.autograd.set_detect_anomaly(True)

from filmingnerf.data import get_data_from_cfg
from filmingnerf.tools.tensor import get_device, move_to
from filmingnerf.opt import get_opts
from filmingnerf.optim import CameraSequencerBase, weight_init, BaseCameraModel
from filmingnerf.camera_init import *
from filmingnerf.renderer.launch_renderer_dnerf import load_layouts, launch_renderer_dnerf

from video_params import check_params
from dnerf_interface import build_nerf, to8b

def run_opt(cfg, dataset, device):
    B = len(dataset)
    T = dataset.seq_len

    loader = DataLoader(dataset, batch_size=int(T), shuffle=False)

    obs_data = move_to(next(iter(loader)), device)

    vis_mask = dataset.get_vis_mask()
    track_ids = dataset.get_track_id()

    tracks_path = os.path.join(cfg.data_root, cfg.seq_name, f"{cfg.seq_name}.npz")
    smpl_model_path = os.path.join(cfg.data_root, 'smpl_tmpl_model.npz')
    layouts, smpl_info, floor_plane, cam_R, cam_t = load_layouts(tracks_path, smpl_model_path, vis_mask, track_ids)

    nerf_data_root, offset_centers, mask_colors = launch_renderer_dnerf(layouts, smpl_info, cfg, viewer=False)
    # continue
    # nerf_config_path = gen_nerf_config_dnerf(nerf_data_root, f'{cfg.seq_name}')

    workspace = os.path.join(nerf_data_root, 'dlogs')
    nerf = build_nerf(nerf_data_root, workspace, device)
    ckpts_path = os.path.join(workspace, 'checkpoints')
    
    if not os.path.exists(ckpts_path) or cfg.overwrite:
        nerf.train()
    return 0



def main():
    cfg = get_opts()
    cfg = check_params(cfg)
    dataset = get_data_from_cfg(cfg)
    device = get_device(0)
    
    run_opt(cfg, dataset, device)


if __name__ == "__main__":
    main()
