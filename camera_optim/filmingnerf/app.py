import gradio as gr

import cv2
import os
import time
import json
import imageio
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] = 'cuda: 0'

import torch
import io
from einops import rearrange
from tqdm import tqdm
import plotly.graph_objects as go
import torchvision.transforms.functional as fn
import torchvision.transforms
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
from data import get_data_from_cfg
from tools.tensor import get_device, move_to
from opt import get_opts

from renderer.launch_renderer import launch_renderer, load_layouts, spherical_camera_pose, gen_nerf_config_hashnerf
from renderer import make_4x4_pose
from renderer.renderer import camera_marker_geometry
from nerf import build_nerf, render_nerf, to8b
from optim import SphericalCameraModel
from preproc.midas import MidasDetector
from camera_init import *

init_params = None
output_image = None

def camera_init(p_x,p_y,p_z,scale,Phi,Theta):
    up = torch.Tensor([0,1,0])
    f_pos = torch.Tensor([p_x,p_y,p_z])
    scale = torch.tensor(scale).float()
    Phi = torch.tensor(Phi).float()
    Theta = torch.tensor(Theta).float()
    f_params = [f_pos, Theta, Phi, scale]
    pose = CalSphericalSpace(f_params, up)
    hwf = [
            h,w,
            0.5*w/np.tan(0.5*xfov),
            0.5*h/np.tan(0.5*yfov)
        ]
    global init_params
    init_params = f_params
    with torch.no_grad(): 
        rgb, depth = nerf.render_by_pose_optim(pose, hwf)
        rgb = rearrange(rgb, '(h w) c -> h w c', h=h)
        
        mask_color_img_tmp  = (mask_color_img/255).to(torch.float) 
        loss = mse_loss(rgb[mask], mask_color_img_tmp[mask])
        img = (rgb.cpu().numpy()*255).astype(np.uint8)
    print('loss:', loss)
    dst = cv2.addWeighted(img, 0.7, rgb_gt.cpu().numpy(), 0.3, 0)
    
    return dst, pose_visualise(pose.cpu())

def pose_optimize():
    camera_model = SphericalCameraModel(init_params).to(device)
    optimizer = torch.optim.Adam(params=camera_model.parameters(), lr=0.01, betas=(0.9, 0.999))
    imgs = []
    depths = []
    print('start camera pose optimizing process......')
    for i_step in tqdm(range(cfg.cam_optim_setps)):
        pose = camera_model()
        hwf = [
                h,w,
                0.5*w/np.tan(0.5*xfov),
                0.5*h/np.tan(0.5*yfov)
            ]
        rgb, depth = nerf.render_by_pose_optim(pose, hwf)
        rgb = rearrange(rgb, '(h w) c -> h w c', h=h)
        depth = rearrange(depth, '(h w) -> h w', h=h)

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
                pre = camera_model.reproject(joints[kj_id],torch.Tensor(mesh_center),hwf)
                
                pre_tmp = pre.int().cpu().numpy()
                gt_tmp = gt.int().cpu().numpy()
                if pre_tmp[1] < h  and pre_tmp[0] < w:
                    joint_img[pre_tmp[1], pre_tmp[0]] = 255
                    joint_img[gt_tmp[1], gt_tmp[0]] = 150

                    kp_loss = kp_loss + torch.abs((gt[:2]-pre[:2])).mean()
        kp_loss = kp_loss / (kjoints2d.shape[0]*(kjoints2d.shape[1]-6))

        loss =  loss_rgb + 0.1 * kp_loss
        optimizer.zero_grad()

        loss.backward()

        if i_step % 10 == 0:
            print(f"Step {i_step}, loss: {loss}")
        
        save_root = os.path.join(cfg.out_dir, cfg.seq_name, str(t))
        os.makedirs(save_root, exist_ok=True)
        img, depth = nerf.render_by_pose(pose)
        depth = depth.cpu().detach().numpy()
        rgb = img.cpu().detach().numpy()
        depth8 = to8b(depth)
        rgb8 = to8b(rgb)
        filename_dep = os.path.join(save_root, str(i_step)+'_depth'+'.png')
        dst_dep = cv2.addWeighted(depth8, 0.7, depth_gt.cpu().numpy(), 0.3, 0)
        
        filename_rgb = os.path.join(save_root, str(i_step)+'_rgb'+'.png')
        dst_rgb = cv2.addWeighted(rgb8, 0.7, rgb_gt.cpu().numpy(), 0.3, 0)

        
        if i_step % 10 == 0:
            joint_img = to8b(joint_img)
            filename = os.path.join(save_root, str(i_step)+'_kpoints'+'.png')
            dst = cv2.addWeighted(joint_img, 0.7, rgb_gt.cpu().numpy(), 0.3, 0)
            imageio.imwrite(filename, dst)
            
            depths.append(dst_dep)
            imgs.append(dst_rgb)
            imageio.imwrite(filename_dep, depth8)
            imageio.imwrite(filename_rgb, rgb8)

        optimizer.step()
    params = camera_model.extract_params()
    x,y,z,scale,phi,theta = params
    saveJson(os.path.join(save_root, f'init_params.json'), params)
    saveJson(os.path.join(save_root, f'init_pose.json'), pose.cpu().tolist())
    imageio.mimwrite(os.path.join(save_root, f'depth_vid.gif'), depths, fps=8)
    imageio.mimwrite(os.path.join(save_root, f'rgb_vid.gif'), imgs, fps=8)
    return imgs[-1],pose_visualise(pose.detach().cpu()),x,y,z,scale,phi,theta

def saveJson(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def pose_visualise(pose,radius=0.2,height=0.3, up="y"):
    cam_verts, cam_faces, _ = camera_marker_geometry(radius, height, up)
    
    for i in range(len(cam_verts)):
        tmp = torch.ones(4).cpu()
        tmp[:3] = torch.Tensor(cam_verts[i]).cpu()
        cam_verts[i] = (pose @ tmp).numpy()[:3]
    
    x = cam_verts[:,0].tolist()
    y = cam_verts[:,1].tolist()
    z = cam_verts[:,2].tolist()
    
    i = cam_faces[:,0].tolist()
    j = cam_faces[:,1].tolist()
    k = cam_faces[:,2].tolist()
    camera_mesh = go.Mesh3d(x=x, y=y, z=z, i=i,j=j,k=k,
                            color='black', opacity=0)
    outline_x = [x[0],x[1],x[1],x[2],x[2],x[3],x[3],x[0],x[0],x[4],x[1],x[4],x[2],x[4], x[3]]
    outline_y = [y[0],y[1],y[1],y[2],y[2],y[3],y[3],y[0],y[0],y[4],y[1],y[4],y[2],y[4], y[3]]
    outline_z = [z[0],z[1],z[1],z[2],z[2],z[3],z[3],z[0],z[0],z[4],z[1],z[4],z[2],z[4], z[3]]
    scatter = go.Scatter3d(x=outline_x, y=outline_y, z=outline_z, mode='lines', 
                           line=dict(width=5))
    
    smpl_verts, _, smpl_faces, _ = layouts
    import copy
    cal_smpl_verts = copy.deepcopy(smpl_verts)
    cal_smpl_verts = cal_smpl_verts[t].squeeze(0)

    transform = torch.eye(4)
    transform[:3,3] = torch.Tensor(mesh_center)
    for i in range(len(cal_smpl_verts)):
        cal_smpl_verts[i][1] = -cal_smpl_verts[i][1]
        cal_smpl_verts[i][2] = -cal_smpl_verts[i][2]
        tmp = torch.ones((4,1))
        tmp[:3, 0] = cal_smpl_verts[i]
        cal_smpl_verts[i] = (transform @ tmp)[:3, 0]

    x,y,z = [],[],[]
    x = cal_smpl_verts[:,0].tolist()
    y = cal_smpl_verts[:,1].tolist()
    z = cal_smpl_verts[:,2].tolist()
    i = smpl_faces[t].squeeze(0)[:,0].tolist()
    j = smpl_faces[t].squeeze(0)[:,1].tolist()
    k = smpl_faces[t].squeeze(0)[:,2].tolist()
    smpl_mesh = go.Mesh3d(x=x, y=y, z=z, i=i,j=j,k=k,
                          color='lightblue', opacity=1)

    fig = go.Figure(data=[scatter, camera_mesh, smpl_mesh])
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=1.25, z=1.25)
)
    fig.update_layout(
        scene_camera=camera,
        scene_aspectmode='cube',
        scene = dict(
            xaxis = dict(nticks=4, range=[-5,5],),
                        yaxis = dict(nticks=4, range=[-5,5],),
                        zaxis = dict(nticks=4, range=[-5,5],),)
                        )
    return fig

def CineUI():
    with gr.Row():
        with gr.Blocks(title='Cinematic Space'):
            with gr.Column():
                gr.Markdown("cinematic core position")
                with gr.Row():
                    p_x = gr.Number(value=0, label='X', min_width=100)
                    p_y = gr.Number(value=0, label='y', min_width=100)
                    p_z = gr.Number(value=0, label='z',min_width=100)
                
                scale = gr.Number(label='Scale', value=5.)
                Phi = gr.Slider(label='Phi', value=0., minimum=0, maximum=np.pi, step=0.01)
                Theta = gr.Slider(label='Theta', value=0., minimum=0, maximum=2*np.pi, step=0.01)
                
                render_button = gr.Button('Render', variant='primary')
                optim_button = gr.Button('Optimize', variant='primary')

        with gr.Blocks():
            with gr.Column():
                out_init_img = gr.Image(label='init pose image', interactive=False)
                visual_plot = gr.Plot()
                out_opti_img = gr.Image(label='optim pose image', interactive=False)

    
    render_button.click(camera_init, inputs=[p_x,p_y,p_z,scale,Phi,Theta], outputs=[out_init_img, visual_plot])
    optim_button.click(pose_optimize, outputs=[out_opti_img, visual_plot,p_x,p_y,p_z,scale,Phi,Theta])

torch.set_default_tensor_type('torch.cuda.FloatTensor')
cfg = get_opts()
dataset = get_data_from_cfg(cfg)
device = get_device(0)

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

t = cfg.t
nerf_data_root, mesh_center, mesh_colors = launch_renderer(layouts, smpl_info, cfg, t, 'train')
nerf_config_path = gen_nerf_config_hashnerf(nerf_data_root, f'{cfg.seq_name}_{t}')
nerf = build_nerf(nerf_config_path, device)
nerf_ckpt_path = os.path.join(nerf_data_root, 'logs')
ckpts = [os.path.join(nerf_ckpt_path, f) for f in sorted(os.listdir(nerf_ckpt_path)) if 'tar' in f]
if len(ckpts) == 0 or cfg.overwrite:
    nerf.train(3000)

with open(os.path.join(nerf_data_root, 'transforms_test.json'), 'r') as f:
    meta = json.load(f)
xfov = meta['camera_angle_x']
yfov = meta['camera_angle_y']

mse_loss = torch.nn.MSELoss()

h, w = cfg.nerf_render_img_hw

depth_gt = obs_data['depth'][t]
rgb_gt = obs_data['rgb'][t]
mask_img = obs_data['mask'][t]/255
mask = obs_data['mask'][t]>0
kjoints2d = obs_data["joints2d"][t]
mapping_ids = smpl_info['mapping_ids']
smpl_joints = smpl_info['joints'][:,t,:,:]
        
print('generate mask color img')
mask_root= dataset.data_sources['mask_root']
track_path = dataset.data_sources['tracks']
seq_name = dataset.seq_name
img_name = dataset.img_names[t]
mask_color_img = np.ones_like(rgb_gt.cpu().numpy())*255
for k in range(len(dataset.track_ids[:])):
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

with gr.Blocks() as demo:
    CineUI()

demo.launch()  


