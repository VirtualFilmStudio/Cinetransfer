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
    # import pdb;pdb.set_trace()
    # smpl_orients_quat = angle_axis_to_quaternion(smpl_orients)
    # smpl_orients_matrix = quaternion_to_rotation_matrix(smpl_orients_quat)
    # motion_matrix = make_4x4_pose(smpl_orients_matrix, smpl_trans).cpu().numpy()
    
    import pickle
    for i in range(len(smpl_orients)):
        smpl_poses = smpl_orients[i].cpu().numpy()
        smpl_trans = smpl_translate[i].cpu().numpy()
        # import pdb;pdb.set_trace()
        out = {
            'smpl_poses':smpl_poses,
            'smpl_trans':smpl_trans,
            'smpl_scaling':1
        }
        with open(f'{cfg.seq_name}_{i}.pkl', 'wb') as file:
            pickle.dump(out, file)

    # from fairmotion.data import bvh

    # BVH_FILENAME = "D:/630_demo/motionDiffusion/ik/joystick/final/out.bvh"
    # motion = bvh.load(BVH_FILENAME)
    
    # from fairmotion.core.motion import Motion

    # # motion_matrix has shape (num_frames, num_joints, 4, 4) where 4x4 is transformation matrix
    # # motion_matrix = motion.to_matrix()

    # translation_matrix = np.zeros((4, 4))
    # translation_matrix[3, :3] = np.array([1, 1, 1])

    # for i in range(len(motion_matrix)):
    #     translated_motion_matrix = motion_matrix[i] + translation_matrix
    #     sliced_motion_matrix = translated_motion_matrix[:,:22]
    #     import pdb;pdb.set_trace()
    #     sliced_motion = Motion.from_matrix(sliced_motion_matrix, motion.skel)

    #     NEW_BVH_FILENAME = f"smpl_{i}.bvh"
    #     bvh.save(sliced_motion, NEW_BVH_FILENAME)




def main():
    cfg = get_opts()
    cfg = check_params(cfg)
    dataset = get_data_from_cfg(cfg)
    device = get_device(0)
    
    run_smpl2bvh(cfg, dataset, device)


if __name__ == "__main__":
    main()
