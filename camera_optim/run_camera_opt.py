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
    cfg.end_id = T
    
    loader = DataLoader(dataset, batch_size=int(T), shuffle=False)

    obs_data = move_to(next(iter(loader)), device)

    vis_mask = dataset.get_vis_mask()
    track_ids = dataset.get_track_id()

    tracks_path = os.path.join(cfg.data_root, cfg.seq_name, f"{cfg.seq_name}.npz")
    smpl_model_path = os.path.join(cfg.data_root, 'smpl_tmpl_model.npz')
    layouts, smpl_info, floor_plane, cam_R, cam_t = load_layouts(tracks_path, smpl_model_path, vis_mask, track_ids)

    nerf_data_root, offset_centers, mask_colors = launch_renderer_dnerf(layouts, smpl_info, cfg, viewer=False)

    workspace = os.path.join(nerf_data_root, 'dlogs')
    nerf = build_nerf(nerf_data_root, workspace, device)
    ckpts_path = os.path.join(workspace, 'checkpoints')
    
    if not os.path.exists(ckpts_path) or cfg.overwrite:
        nerf.train()

    ckpts = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if 'pth' in f]
    nerf.load_ckpt(ckpts[-1])
    nerf.model.to(device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    out_imgs = []
    out_poses = []

    def openJson(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    
    save_root = os.path.join(cfg.out_dir, cfg.seq_name, str(0), 'init_pose.json')
    if os.path.exists(save_root):
        init_pose = openJson(save_root)
    else:
        from run_camera_vis import transform_pyrender, transform_slam
        camera_poses_w2c = make_4x4_pose(cam_R, cam_t)
        camera_poses_c2w = torch.linalg.inv(camera_poses_w2c)
        camera_poses_c2w = transform_pyrender(camera_poses_w2c)

        camera_poses_c2w = transform_slam(torch.Tensor(camera_poses_c2w))
        camera_poses_c2w[:,3,:] = np.array([0,0,0,1])
        
        init_pose = camera_poses_c2w[0]
    init_pose = torch.Tensor(init_pose).to(device)


    with open(os.path.join(nerf_data_root, 'transforms_test.json'), 'r') as f:
        meta = json.load(f)
    xfov = meta['camera_angle_x']
    yfov = meta['camera_angle_y']

    camera_model = CameraSequencerBase().to(device)

    save_fig_root = os.path.join(cfg.out_dir, cfg.seq_name, 'optim')
    os.makedirs(save_fig_root, exist_ok=True)
    save_fig_path = os.path.join(save_fig_root, f'init_func.png')
    camera_model.show_fig(T, save=True, save_path=save_fig_path)
    
    optimizer = torch.optim.Adam(params=camera_model.parameters(), lr=0.001, betas=(0.9, 0.999))
    mse_loss = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    losses = torch.zeros(T)
    print('start camera pose optimizing process......')
    imgs = [0 for i in range(T)]
    poses = [0 for i in range(T)]
    poses[0] = init_pose.tolist()
    for t in range(T)[cfg.t+1:cfg.end_id]:
        mesh_center = offset_centers[t]
        mesh_colors = mask_colors[t]
        mesh_center = torch.Tensor(mesh_center)
        mesh_colors = torch.Tensor(mesh_colors)

        with open(os.path.join(nerf_data_root, 'transforms_test.json'), 'r') as f:
            meta = json.load(f)
        xfov = float(meta['camera_angle_x'])
        yfov = float(meta['camera_angle_y'])

        rgb_gt = obs_data['rgb'][t]
        mask_img = obs_data['mask'][t]/255
        mask = obs_data['mask'][t]>0
        kjoints2d = obs_data["joints2d"][t]
        mapping_ids = smpl_info['mapping_ids']
        smpl_joints = smpl_info['joints'][:,t,:,:]
        h, w, _ = rgb_gt.shape 

        mask_root= dataset.data_sources['mask_root']
        track_path = dataset.data_sources['tracks']
        seq_name = dataset.seq_name
        img_name = dataset.img_names[t]
        mask_color_img = np.ones_like(rgb_gt.cpu().numpy())*255
        skip_track = []
        end = min(cfg.track_sid+len(mesh_colors), cfg.track_eid)
        for k in range(len(dataset.track_ids[cfg.track_sid:end])):
            t_id = dataset.track_ids[k]
            color = mesh_colors[k]
            kp_path = os.path.join(track_path, t_id, f"{img_name}_keypoints.json")
            if not os.path.exists(kp_path):
                skip_track.append(k)
                continue
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
        epochs = 80
        
        for i_step in tqdm(range(epochs)):
            time_step = 1/(T-1)
            pose = camera_model(init_pose, t*time_step)
            poses[t] = pose.tolist()
            ngp_pose = camera_model.nerf_matrix_to_ngp(pose)

            hwf = [
                    h,w,
                    0.5*w/np.tan(0.5*xfov),
                    0.5*h/np.tan(0.5*yfov)
                ]
            
            rgb, depth = nerf.render_optim(ngp_pose, hwf, t*time_step)
            rgb = rearrange(rgb, '(h w) c -> h w c', h=h)

            depth = rearrange(depth, '(h w) -> h w', h=h)
            mask_dep = depth > 0
            depth[mask_dep] = (depth[mask_dep] - near) / (far - near) 

            mask_color_img_tmp  = (mask_color_img/255).to(torch.float) 
            loss_rgb = mse_loss(rgb[mask], mask_color_img_tmp[mask])

            rgb_copy = rgb.cpu().detach().numpy()

            joint_img = np.zeros_like(rgb_copy)
            kp_loss = torch.tensor(0)
            for n_tracks in range(smpl_joints.shape[0])[cfg.track_sid:cfg.track_eid]:
                joints = smpl_joints[n_tracks][mapping_ids]
                joints_gt = kjoints2d[n_tracks]
                
                if n_tracks in skip_track:
                    continue

                for kj_id in range(len(joints)):
                    if kj_id in [22,23,24,20,21,19]:
                        continue
                    gt = (joints_gt[kj_id] / cfg.downsample).int()
                    pre = camera_model.reproject(joints[kj_id],mesh_center,hwf)
                    
                    pre_tmp = pre.int().cpu().numpy()
                    gt_tmp = gt.int().cpu().numpy()
                    if pre_tmp[1] < h and pre_tmp[1] > 0 and pre_tmp[0] < w and pre_tmp[0] >0:
                        joint_img[pre_tmp[1], pre_tmp[0]] = 255
                        kp_loss = kp_loss + torch.abs((gt[:2]-pre[:2])).mean()
                    if gt_tmp[1] < h and gt_tmp[0] < w:
                        joint_img[gt_tmp[1], gt_tmp[0]] = 150
            
            kp_loss = kp_loss / (kjoints2d.shape[0]*(kjoints2d.shape[1]-6))
            loss = loss_rgb + kp_loss

            if i_step % 10 == 0:
                print(f"Step {t}, loss: {loss}")
                save_root = os.path.join(cfg.out_dir, cfg.seq_name, 'optim', str(t))
                os.makedirs(save_root, exist_ok=True)
                depth = depth.cpu().detach().numpy()
                rgb = rgb.cpu().detach().numpy()
                rgb8 = to8b(rgb)
                joint_img = to8b(joint_img)
                filename = os.path.join(save_root, str(i_step)+"_"+str(t)+'_kpoints'+'.png')
                dst = cv2.addWeighted(joint_img, 0.7, rgb_gt.cpu().numpy(), 0.3, 0)
                imageio.imwrite(filename, dst)
                filename = os.path.join(save_root, str(i_step)+"_"+str(t)+'_rgb'+'.png')
                imageio.imwrite(filename, rgb8)
                dst = cv2.addWeighted(rgb8, 0.7, rgb_gt.cpu().numpy(), 0.3, 0)
                imgs[t] = dst
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        save_fig_path = os.path.join(cfg.out_dir, cfg.seq_name, 'optim', f'{t}_func.png')
        camera_model.show_fig(T, save=True, save_path=save_fig_path)
    
    save_fig_path = os.path.join(cfg.out_dir, cfg.seq_name, 'optim', f'{i_step}_func.png')
    camera_model.show_fig(T, save=True, save_path=save_fig_path)
    save_root = os.path.join(cfg.out_dir, cfg.seq_name)
    os.makedirs(save_root, exist_ok=True)
    imageio.mimwrite(os.path.join(save_root, f'vid_final.gif'), imgs[1:], fps=30)

    
    out_poses = poses
    with open(os.path.join(save_root, 'optim_cams.json'), "w") as f:
        json.dump(out_poses, f, indent=4)

    long_poses = []
    long_poses.append(poses[0])
    init_pose= torch.Tensor(poses[0]).to(device)
    T = 500
    time_step = 1/(T-1)
    for t in range(500)[cfg.t+1:]:
        pose = camera_model(init_pose, t*time_step)
        long_poses.append(pose.tolist())
        init_pose = torch.Tensor(pose.tolist()).to(device)
    with open(os.path.join(save_root, 'optim_cams_long.json'), "w") as f:
        json.dump(long_poses, f, indent=4)


def main():
    cfg = get_opts()
    cfg = check_params(cfg)
    dataset = get_data_from_cfg(cfg)
    device = get_device(0)
    
    run_opt(cfg, dataset, device)


if __name__ == "__main__":
    main()
