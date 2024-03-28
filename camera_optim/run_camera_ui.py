import gradio as gr
import plotly.graph_objects as go

import cv2
import os
import time
import json
import imageio
import numpy as np
from PIL import Image

import torch
from einops import rearrange
from tqdm import tqdm
import torchvision.transforms.functional as fn
import torchvision.transforms
from torch.utils.data import DataLoader

from filmingnerf.data import get_data_from_cfg
from filmingnerf.tools.tensor import get_device, move_to
from filmingnerf.opt import get_opts
from filmingnerf.optim import DoubleTajCameraModel, SphericalCameraModel, BaseCameraModel
from filmingnerf.camera_init import CalSphericalSpace, CalSphericalSpaceNGP
from filmingnerf.renderer.launch_renderer_dnerf import load_layouts, launch_renderer_dnerf
from filmingnerf.renderer.renderer import camera_marker_geometry

from dnerf_interface import build_nerf, to8b
from video_params import check_params

init_pose = None
init_params = None
output_image = None

def camera_init(p_x,p_y,p_z,scale,Phi,Theta):
    up = torch.Tensor([0,-1,0])
    f_pos = torch.Tensor([p_x,p_y,p_z])
    scale = torch.tensor(scale).float()
    Phi = torch.tensor(Phi).float()
    Theta = torch.tensor(Theta).float()
    f_params = [f_pos, Theta, Phi, scale]
    norm_pose, cam_pos = CalSphericalSpace(f_params, torch.Tensor([0,1,0]))
    ngp_pose = CalSphericalSpaceNGP(f_params, torch.Tensor([0,-1,0]))
    hwf = [
            h,w,
            0.5*w/np.tan(0.5*xfov),
            0.5*h/np.tan(0.5*yfov)
        ]
    global init_params
    global init_pose
    init_pose = norm_pose
    init_params = [f_pos, Theta, Phi, scale]
    with torch.no_grad(): 
        camera_model = BaseCameraModel(xfov, yfov).to(device)
        pose = camera_model(norm_pose)
        ngp_pose = camera_model.nerf_matrix_to_ngp(pose)

        time_step = 1/(T-1)
        rgb, depth = nerf.render_optim(ngp_pose, hwf, 0*time_step)
        rgb = rearrange(rgb, '(h w) c -> h w c', h=h)
        mask_color_img_tmp  = (mask_color_img/255).to(torch.float) 
        loss = mse_loss(rgb[mask], mask_color_img_tmp[mask])
        img = (rgb.cpu().numpy()*255).astype(np.uint8)
    print('loss:', loss)
    dst = cv2.addWeighted(img, 0.7, rgb_gt.cpu().numpy(), 0.3, 0)

    tmp_path = os.path.join(cfg.out_dir, cfg.seq_name)
    os.makedirs(tmp_path, exist_ok=True)
    cv2.imwrite(os.path.join(tmp_path, 'cache_first_pose.png'), dst)
    
    return dst, pose_visualise(pose.cpu())

def pose_optimize():
    camera_model = BaseCameraModel(xfov, yfov).to(device)
    optimizer = torch.optim.Adam(params=camera_model.parameters(), lr=0.01, betas=(0.9, 0.999))
    imgs = []
    print('start camera pose optimizing process......')
    for i_step in tqdm(range(cfg.cam_optim_setps)):
        pose = camera_model(init_pose)
        ngp_pose = camera_model.nerf_matrix_to_ngp(pose)
        hwf = [
                h,w,
                0.5*w/np.tan(0.5*xfov),
                0.5*h/np.tan(0.5*yfov)
            ]
        time_step = 1/(T-1)
        rgb, depth = nerf.render_optim(ngp_pose, hwf, cfg.t*time_step)
        rgb = rearrange(rgb, '(h w) c -> h w c', h=h)
        depth = rearrange(depth, '(h w) -> h w', h=h)

        mask_color_img_tmp  = (mask_color_img/255).to(torch.float) 
        loss_rgb = mse_loss(rgb[mask], mask_color_img_tmp[mask])

        rgb_copy = rgb.cpu().detach().numpy()
        joint_img = np.zeros_like(rgb_copy)
        kp_loss = torch.tensor(0)
        for n_tracks in range(smpl_joints.shape[0])[cfg.track_sid:cfg.track_eid]:
            joints = smpl_joints[n_tracks][mapping_ids]
            joints_gt = kjoints2d[n_tracks]
            
            for kj_id in range(len(joints)):
                if kj_id in [22,23,24,20,21,19]:
                    continue
                gt = (joints_gt[kj_id] / cfg.downsample).int()
                pre = camera_model.reproject(joints[kj_id],torch.Tensor(mesh_center),hwf)
                
                pre_tmp = pre.int().cpu().numpy()
                gt_tmp = gt.int().cpu().numpy()
                if pre_tmp[1] < h and pre_tmp[1] > 0  and pre_tmp[0] < w and pre_tmp[0] > 0  :
                    joint_img[pre_tmp[1], pre_tmp[0]] = 255
                    joint_img[gt_tmp[1], gt_tmp[0]] = 150
                if gt_tmp[1] < h and gt_tmp[1] > 0  and gt_tmp[0] < w and gt_tmp[0] > 0:
                    kp_loss = kp_loss + torch.abs((gt[:2]-pre[:2])).mean()
        kp_loss = kp_loss.mean()

        loss = loss_rgb + kp_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_step % 10 == 0:
            print(f"Step {i_step}, loss: {loss}")
        
        save_root = os.path.join(cfg.out_dir, cfg.seq_name, str(t))
        os.makedirs(save_root, exist_ok=True)

        rgb = rgb.cpu().detach().numpy()
        rgb8 = to8b(rgb)
        
        filename_rgb = os.path.join(save_root, str(i_step)+'_rgb'+'.png')
        dst_rgb = cv2.addWeighted(rgb8, 0.7, rgb_gt.cpu().numpy(), 0.3, 0)

        
        if i_step % 10 == 0:
            joint_img = to8b(joint_img)
            filename = os.path.join(save_root, str(i_step)+'_kpoints'+'.png')
            dst = cv2.addWeighted(joint_img, 0.7, rgb_gt.cpu().numpy(), 0.3, 0)
            imageio.imwrite(filename, dst)
            

            imgs.append(dst_rgb)
            imageio.imwrite(filename_rgb, rgb8)

        
    saveJson(os.path.join(save_root, f'init_pose.json'), pose.cpu().tolist())
    imageio.mimwrite(os.path.join(save_root, f'rgb_vid.gif'), imgs, fps=8)
    return imgs[-1],pose_visualise(camera_model.T.detach().cpu())

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
    cal_smpl_verts = cal_smpl_verts[t]

    data = [scatter, camera_mesh]
    for k in range(cal_smpl_verts.shape[0]):
        transform = torch.eye(4)
        transform[:3,3] = torch.Tensor(mesh_center)

        for i in range(len(cal_smpl_verts[k])):
            cal_smpl_verts[k][i][1] = -cal_smpl_verts[k][i][1]
            cal_smpl_verts[k][i][2] = -cal_smpl_verts[k][i][2]
            tmp = torch.ones((4,1))
            tmp[:3, 0] = cal_smpl_verts[k][i]
            cal_smpl_verts[k][i] = (transform @ tmp)[:3, 0]

        x,y,z = [],[],[]
        x = cal_smpl_verts[k][:, 0].tolist()
        y = cal_smpl_verts[k][:, 1].tolist()
        z = cal_smpl_verts[k][:, 2].tolist()
  
        i = smpl_faces[t][:,0].tolist()
        j = smpl_faces[t][:,1].tolist()
        k = smpl_faces[t][:,2].tolist()

        smpl_mesh = go.Mesh3d(x=x, y=y, z=z, i=i,j=j,k=k,
                            color='lightblue', opacity=1)
        data.append(smpl_mesh)

    fig = go.Figure(data=data)
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=1.25, z=1.25)
    )
    fig.update_layout(
        scene_camera=camera,
        scene_aspectmode='cube',
        scene = dict(xaxis = dict(nticks=4, range=[-5,5],),
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
                out_init_img = gr.Image(label='init pose image', interactive=False, show_download_button=True)
                visual_plot = gr.Plot()
                out_opti_img = gr.Image(label='optim pose image', interactive=False)
    render_button.click(camera_init, inputs=[p_x,p_y,p_z,scale,Phi,Theta], outputs=[out_init_img, visual_plot])
    optim_button.click(pose_optimize, outputs=[out_opti_img, visual_plot])

if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cfg = get_opts()
    cfg = check_params(cfg)
    dataset = get_data_from_cfg(cfg)
    device = get_device(0)

    B = len(dataset)
    T = dataset.seq_len
    cam_data= dataset.get_camera_data()

    loader = DataLoader(dataset, batch_size=int(T), shuffle=False)

    obs_data = move_to(next(iter(loader)), device)

    vis_mask = dataset.get_vis_mask()
    track_ids = dataset.get_track_id()

    tracks_path = os.path.join(cfg.data_root, cfg.seq_name, f"{cfg.seq_name}.npz")
    smpl_model_path = os.path.join(cfg.data_root, 'smpl_tmpl_model.npz')
    layouts, smpl_info, _, _, _ = load_layouts(tracks_path, smpl_model_path, vis_mask, track_ids)

    t = cfg.t
    nerf_data_root, offset_centers, mask_colors = launch_renderer_dnerf(layouts, smpl_info, cfg, viewer=False)

    workspace = os.path.join(nerf_data_root, 'dlogs')
    nerf = build_nerf(nerf_data_root, workspace, device)
    ckpts_path = os.path.join(workspace, 'checkpoints')

    if not os.path.exists(ckpts_path) or cfg.overwrite:
        nerf.train()

    ckpts = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if 'pth' in f]
    nerf.load_ckpt(ckpts[-1])
    nerf.model.to(device)

    with open(os.path.join(nerf_data_root, 'transforms_test.json'), 'r') as f:
        meta = json.load(f)
    xfov = meta['camera_angle_x']
    yfov = meta['camera_angle_y']

    mse_loss = torch.nn.MSELoss()

    h, w = cfg.nerf_render_img_hw

    mesh_center = offset_centers[t]
    mesh_colors = mask_colors[t]
    mesh_center = torch.Tensor(mesh_center)
    mesh_colors = torch.Tensor(mesh_colors)

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
    start = cfg.track_sid
    end = min(cfg.track_sid+len(mesh_colors), cfg.track_eid)
    for k in range(len(dataset.track_ids[cfg.track_sid:end])):
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


