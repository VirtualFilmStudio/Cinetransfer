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
torch.autograd.set_detect_anomaly(True)
from data import get_data_from_cfg
from tools.tensor import get_device, move_to
from opt import get_opts

from renderer.launch_renderer import launch_renderer, load_layouts, spherical_camera_pose, gen_nerf_config_hashnerf
from renderer import make_4x4_pose
from nerf import build_nerf, render_nerf, to8b
from optim import BaseCameraModel, IntrinsicsCameraModel
from preproc.midas import MidasDetector
from camera_init import *

def run_opt(cfg, dataset, device):
    B = len(dataset)
    T = dataset.seq_len
    cam_data= dataset.get_camera_data()

    camera_poses_w2c = make_4x4_pose(cam_data['cam_R'], cam_data['cam_t'])
    camera_poses_c2w = torch.linalg.inv(camera_poses_w2c)

    loader = DataLoader(dataset, batch_size=int(T), shuffle=False)

    obs_data = move_to(next(iter(loader)), device)

    vis_mask = dataset.get_vis_mask()
    track_ids = dataset.get_track_id()

    tracks_path = os.path.join(cfg.data_root, cfg.seq_name, f"{cfg.seq_name}.npz")
    smpl_model_path = os.path.join(cfg.data_root, 'smpl_tmpl_model.npz')
    layouts, smpl_info = load_layouts(tracks_path, smpl_model_path, vis_mask, track_ids)

    out_imgs = []
    out_poses = []

    def openJson(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
            return data
    save_root = os.path.join(cfg.out_dir, cfg.seq_name, str(0), 'init_pose.json')
    init_pose = openJson(save_root)
    init_pose = torch.Tensor(init_pose).to(device)
    for t in range(T)[1::2]:
        nerf_data_root, mesh_center, mesh_colors = launch_renderer(layouts, smpl_info, cfg, t, 'train', viewer=False)
        mesh_center = torch.Tensor(mesh_center)
        # import pdb;pdb.set_trace()
        # continue
        nerf_config_path = gen_nerf_config_hashnerf(nerf_data_root, f'{cfg.seq_name}_{t}')
        nerf = build_nerf(nerf_config_path, device)
        nerf_ckpt_path = os.path.join(nerf_data_root, 'logs')
        ckpts = [os.path.join(nerf_ckpt_path, f) for f in sorted(os.listdir(nerf_ckpt_path)) if 'tar' in f]
        if len(ckpts) == 0 or cfg.overwrite:
            nerf.train(3000)

        with open(os.path.join(nerf_data_root, 'transforms_test.json'), 'r') as f:
            meta = json.load(f)
        xfov = float(meta['camera_angle_x'])
        yfov = float(meta['camera_angle_y'])

        camera_model = BaseCameraModel(xfov, yfov).to(device)
        apply_midas = MidasDetector()
        optimizer = torch.optim.Adam(params=camera_model.parameters(), lr=0.01, betas=(0.9, 0.999))
        mse_loss = torch.nn.MSELoss()

        depth_gt = obs_data['depth'][t]
        rgb_gt = obs_data['rgb'][t]
        mask_img = obs_data['mask'][t]/255
        mask = obs_data['mask'][t]>0
        kjoints2d = obs_data["joints2d"][t]
        mapping_ids = smpl_info['mapping_ids']
        smpl_joints = smpl_info['joints'][:,t,:,:]
        h, w, _ = rgb_gt.shape 

        print('generate mask color img')
        mask_root= dataset.data_sources['mask_root']
        track_path = dataset.data_sources['tracks']
        seq_name = dataset.seq_name
        img_name = dataset.img_names[t]
        mask_color_img = np.ones_like(rgb_gt.cpu().numpy())*255
        for k in range(len(dataset.track_ids[:2])):
            t_id = dataset.track_ids[k]
            color = mesh_colors[k]
            kp_path = os.path.join(track_path, t_id, f"{img_name}_keypoints.json")
            with open(kp_path) as keypoint_file:
                data = json.load(keypoint_file)
            person_data = data["people"][0]
            mask_file_name = person_data["mask_path"].split('/')[-1]
            mask_file_path = os.path.join(mask_root, mask_file_name)
            mask_img_p = imageio.imread(mask_file_path)
            mask_img_p = cv2.resize(mask_img_p, (w, h))
            color_mask = mask_img_p > 0

            mask_color_img[color_mask] = color.cpu().numpy()
        mask_color_img = torch.tensor(mask_color_img).to(device)
        start=1
        near = 1
        far = 30

        # if t == 0:
        #     init_pose = spherical_sampling(10, [0,0,0], 4)[1]
        # with torch.no_grad(): 
        #     pose = torch.FloatTensor(init_pose).cuda()
        #     pose, hwf = camera_model(pose, h, w)
        #     rgb, depth = nerf.render_by_pose_optim(pose, hwf)
        #     rgb = rearrange(rgb, '(h w) c -> h w c', h=h)
            
        #     img = (rgb.cpu().numpy()*255).astype(np.uint8)

        # joint_img = np.zeros_like(img)
        # for n_tracks in range(smpl_joints.shape[0]):
        #     joints = smpl_joints[n_tracks][mapping_ids]
        #     joints_gt = kjoints2d[n_tracks]
        #     for kj_id in range(len(joints)):
        #         if kj_id in [22,23,24, 20,21,19]:
        #             continue
        #         gt = joints_gt[kj_id].int().cpu().numpy()
        #         joint2d = camera_model.reproject(joints[kj_id], mesh_center, w, h)
        #         joint2d = joint2d.int().cpu().numpy()
        #         if joint2d[1] < h and joint2d[0] < w:
        #             joint_img[joint2d[1], joint2d[0]] = 255
        #         # if gt[1] < h and gt[0] < w:
        #         joint_img[int(gt[1]//cfg.downsample), int(gt[0]//cfg.downsample)] = 150
        #         print(joint2d[1], joint2d[0])
        #         print(gt[1]//cfg.downsample, gt[0]//cfg.downsample)
        #         # import pdb;pdb.set_trace()  
        # dst = cv2.addWeighted(joint_img, 0.7, img, 0.3, 0)
        # cv2.imshow('', dst)
        # cv2.waitKey(0)
        # import pdb;pdb.set_trace()                



        imgs = []
        depths = []
        print('start camera pose optimizing process......')
        for i_step in tqdm(range(cfg.cam_optim_setps)):
        # for i_step in tqdm(range(10)):
            pose = init_pose
            pose, hwf = camera_model(pose, h, w)
            
            rgb, depth = nerf.render_by_pose_optim(pose, hwf)
            rgb = rearrange(rgb, '(h w) c -> h w c', h=h)
            depth = rearrange(depth, '(h w) -> h w', h=h)
            mask_dep = depth > 0
            depth[mask_dep] = (depth[mask_dep] - near) / (far - near) 

            mask_color_img_tmp  = (mask_color_img/255).to(torch.float) 
            loss_rgb = mse_loss(rgb[mask], mask_color_img_tmp[mask])

            rgb_copy = rgb.cpu().detach().numpy()
            joint_img = np.zeros_like(rgb_copy)
            kp_loss = torch.tensor(0)
            for n_tracks in range(smpl_joints.shape[0]):
                joints = smpl_joints[n_tracks][mapping_ids]
                joints_gt = kjoints2d[n_tracks]
                
                for kj_id in range(len(joints)):
                    if kj_id in [22,23,24,20,21,19]:
                        continue
                    gt = (joints_gt[kj_id] / cfg.downsample).int()
                    pre = camera_model.reproject(joints[kj_id],mesh_center,w,h)
                    
                    pre_tmp = pre.int().cpu().numpy()
                    gt_tmp = gt.int().cpu().numpy()
                    if pre_tmp[1] < h  and pre_tmp[0] < w:
                        joint_img[pre_tmp[1], pre_tmp[0]] = 255
                        joint_img[gt_tmp[1], gt_tmp[0]] = 150
                    # print(i_step, pre_tmp[1], pre_tmp[0])
                    # print(gt_tmp[1], gt_tmp[0])
                        kp_loss = kp_loss + torch.abs((gt[:2]-pre[:2])).mean()
            kp_loss = kp_loss / (kjoints2d.shape[0]*(kjoints2d.shape[1]-6))
  
            loss = loss_rgb + 0.5 * kp_loss

            # loss = loss_rgb
            # import pdb;pdb.set_trace()
            optimizer.zero_grad()


            loss.backward()

            if i_step % 10 == 0:
                print(f"Step {i_step}, loss: {loss}")
            if i_step % 10 == 0:
                save_root = os.path.join(cfg.out_dir, cfg.seq_name, str(t))
                os.makedirs(save_root, exist_ok=True)
                img, depth = nerf.render_by_pose(pose)
                depth = depth.cpu().detach().numpy()
                rgb = img.cpu().detach().numpy()
                depth8 = to8b(depth)
                rgb8 = to8b(rgb)
                joint_img = to8b(joint_img)
                filename = os.path.join(save_root, str(i_step)+'_kpoints'+'.png')
                dst = cv2.addWeighted(joint_img, 0.7, rgb_gt.cpu().numpy(), 0.3, 0)
                imageio.imwrite(filename, dst)
                filename = os.path.join(save_root, str(i_step)+'_depth'+'.png')
                imageio.imwrite(filename, depth8)
                dst = cv2.addWeighted(depth8, 0.7, depth_gt.cpu().numpy(), 0.3, 0)
                depths.append(dst)
                filename = os.path.join(save_root, str(i_step)+'_rgb'+'.png')
                imageio.imwrite(filename, rgb8)
                dst = cv2.addWeighted(rgb8, 0.7, rgb_gt.cpu().numpy(), 0.3, 0)
                imgs.append(dst)

            optimizer.step()
        
        init_pose = pose.detach()

        imageio.mimwrite(os.path.join(save_root, 'depth_vid.gif'), depths, fps=8)
        imageio.mimwrite(os.path.join(save_root, 'rgb_vid.gif'), imgs, fps=8)
        # import pdb;pdb.set_trace()
        pose, hwf = camera_model(init_pose, h, w)
        
        img, depth = nerf.render_by_pose(pose)

        depth = depth.cpu().detach().numpy()
        rgb = img.cpu().detach().numpy()
        
        out_imgs.append(rgb)
        out_poses.append(pose.cpu().tolist())
    
    save_root = os.path.join(cfg.out_dir, cfg.seq_name)
    os.makedirs(save_root, exist_ok=True)
    imageio.mimwrite(os.path.join(save_root, 'final_vid.gif'), out_imgs, fps=8)
    
    with open(os.path.join(save_root, 'optim_cams.json'), "w") as f:
        json.dump(out_poses, f, indent=4)



def show_grad(module):
    for name, parms in module.named_parameters():
        print('-->name:', name)
        print('-->para', parms)
        print('-->grad_requirs', parms.requires_grad)
        print('-->grad_value:', parms.grad)
        print('===')


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img


def convert_nerf_results(rgb, depth, h):
    rgb_pred = rearrange(rgb.cpu().numpy(), '(h w) c -> h w c', h=h)
    rgb_pred = (rgb_pred*255).astype(np.uint8)
    depth = depth2img(rearrange(depth.cpu().numpy(), '(h w) -> h w', h=h))
    return rgb_pred, depth


def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cfg = get_opts()
    dataset = get_data_from_cfg(cfg)
    device = get_device(0)
    run_opt(cfg, dataset, device)


if __name__ == "__main__":
    main()
    # from renderer import test
    # test.test()