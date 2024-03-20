import torch
import os
import numpy as np
import time
import torch.nn as nn
import pickle
import imageio
from tqdm import tqdm, trange

from torch_ngp import *

class DNerfNGP:

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.args.fp16 = True
        self.args.cuda_ray = True
        self.args.preload = True
        
        seed_everything(self.args.seed)
        self._create_nerf()

    def _create_nerf(self):
        self.model = NeRFNetwork(
            bound=self.args.bound,
            cuda_ray=self.args.cuda_ray,
            density_scale=1,
            min_near=self.args.min_near,
            density_thresh=self.args.density_thresh,
            bg_radius=self.args.bg_radius,
        )
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.optimizer = lambda model: torch.optim.Adam(model.get_params(self.args.lr, self.args.lr_net), betas=(0.9, 0.99), eps=1e-15)
        self.scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / self.args.iters, 1))
            
    def train(self):
        train_loader = NeRFDataset(self.args, device=self.device, type='train').dataloader()
        trainer = Trainer('ngp', self.args, self.model, device=self.device, workspace=self.args.workspace, optimizer=self.optimizer, criterion=self.criterion, ema_decay=0.95, fp16=self.args.fp16, lr_scheduler=self.scheduler, scheduler_update_every_step=True, metrics=[PSNRMeter()], use_checkpoint=self.args.ckpt, eval_interval=50)

        valid_loader = NeRFDataset(self.args, device=self.device, type='val', downscale=1).dataloader()

        max_epoch = np.ceil(self.args.iters / len(train_loader)).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)

        test_loader = NeRFDataset(self.args, device=self.device, type='test').dataloader()
        if test_loader.has_gt:
            trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            
        trainer.test(test_loader, write_video=True) # test and save video

    def load_ckpt(self, ckpt):
        checkpoint_dict = torch.load(ckpt, map_location=self.device)
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            print(f"[INFO] loaded model in {ckpt}.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        print("[INFO] loaded model.")
        if len(missing_keys) > 0:
            print(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"[WARN] unexpected keys: {unexpected_keys}")   


        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']


    def render_optim(self, pose, render_hwf, frame_time):
        h, w, fx, fy = render_hwf
        K = np.array([
            [fx, 0, 0.5*w],
            [0, fy, 0.5*h],
            [0, 0, 1]
        ])
        rays_o, rays_d = get_rays(h, w, K, torch.Tensor(pose))
        coords = torch.stack(torch.meshgrid(torch.linspace(0, h-1, h), torch.linspace(0, w-1, w)), -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1,2]).long() # (H * W, 2)
        rays_o = rays_o[coords[:, 0], coords[:, 1]].unsqueeze(0)
        rays_d = rays_d[coords[:, 0], coords[:, 1]].unsqueeze(0)
        time = torch.zeros((1,1))
        time[0][0] = frame_time
        
        outputs = self.model.render(rays_o, rays_d, time, staged=False, bg_color=None, perturb=True, force_all_rays=True, **vars(self.args))
        return outputs['image'], outputs['depth']    
            
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def build_nerf(data_root, worksapce, device):
    parser = config_parser()
    args = parser.parse_args(args=[data_root, '--workspace', worksapce, '-O', '--bound', str(1.0), '--scale', str(0.8), '--dt_gamma', str(0)])
    Dnerf = DNerfNGP(args, device)
    return Dnerf

def render_nerf():
    pass