import numpy as np
import torch
import os
import torch
import json
import numpy as np
import trimesh
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import pyrender
from pyrender import RenderFlags

from .renderer import Renderer, Transform

from .smpl_model import run_smpl, SmplModel, smpl_to_geometry
from .geometry_np.camera import camera_look_at, spherical_sample
from .geometry.mesh import make_mesh, get_scene_bb

def load_smpl(tracks_path, smpl_model_path, vis_mask, track_ids):
   data = np.load(tracks_path)
   res = {}
   for name in data.files:
      res.update({name: torch.tensor(data[name])})
   # import pdb;pdb.set_trace()
   B, T, _ = res["trans"].shape

   smpl_model = SmplModel(smpl_model_path, batch_size=B*T)

   with torch.no_grad():
      world_smpl = run_smpl(
         smpl_model, 
         res["trans"],
         res["root_orient"],
         res["pose_body"],
         res.get("betas", None)
      )

   hand_orient = torch.zeros_like(res['root_orient'])
   joints_pose = torch.cat((res['root_orient'],res["pose_body"], hand_orient, hand_orient), axis=2)
   # joints_pose = joints_pose.view(B, T, 24, 3)

   # joints_trans = world_smpl['joints'][:,:,:24]

   
   return joints_pose, res["trans"]

def load_layouts(tracks_path, smpl_model_path, vis_mask, track_ids):
   data = np.load(tracks_path)
   res = {}
   for name in data.files:
      res.update({name: torch.tensor(data[name])})
   # import pdb;pdb.set_trace()
   B, T, _ = res["trans"].shape

   smpl_model = SmplModel(smpl_model_path, batch_size=B*T)
   
   # import pickle
   # smpl_poses = np.zeros((T, 22))
   # for i in range(len(res["trans"])):
   #    # import pdb;pdb.set_trace()
   #    poses = torch.cat((res['root_orient'][i],res["pose_body"][i]), axis=1)
   #    smpl_poses = poses.cpu().numpy()
   #    smpl_trans = res["trans"][i].cpu().numpy()
   #    out = {
   #       'smpl_poses':smpl_poses,
   #       'smpl_trans':smpl_trans,
   #       'smpl_scaling':[1]
   #    }
   #    with open(f'./smpl_{i}.pkl', 'wb') as file:
   #       pickle.dump(out, file)
   

   with torch.no_grad():
      world_smpl = run_smpl(
         smpl_model, 
         res["trans"],
         res["root_orient"],
         res["pose_body"],
         res.get("betas", None)
      )
   
   smpl_geometries = smpl_to_geometry(
      world_smpl['vertices'],world_smpl['faces'],
      vis_mask, track_ids)
   # import pdb;pdb.set_trace()
   return smpl_geometries, world_smpl, res['floor_plane'], res["cam_R"][0], res["cam_t"][0]


def spherical_camera_pose(n_samples, center, radius):
   sample_pos = spherical_sample(n_samples)
   up = np.array([0,1,0])
   cam_poses = []
   for pos in sample_pos:
      # import pdb;pdb.set_trace()
      src_pos = [i*radius for i in pos]
      camera_pose = camera_look_at(src_pos, np.array(center), up)
      cam_poses.append(camera_pose)
   return cam_poses


def prep_shared_geometry(verts, faces, colors):
   """
   :param verts (B, V, 3)
   :param faces (F, 3)
   :param colors (B, 4)
   """
   B, V, _ = verts.shape
   F, _ = faces.shape
   colors = colors.unsqueeze(1).expand(B, V, -1)[..., :3]
   faces = faces.unsqueeze(0).expand(B, F, -1)
   return verts, faces, colors


def launch_renderer(layouts, smpl_info, cfg, t, split='train', viewer=False):
   save_root= os.path.join(cfg.data_root+'/', cfg.seq_name+'/', 'nerf/', str(t))
   os.makedirs(save_root, exist_ok=True)
   img_save_path =  os.path.join(save_root, split)
   os.makedirs(img_save_path, exist_ok=True)  
   out_flags = os.listdir(img_save_path)
   mapping_ids = smpl_info['mapping_ids']
   smpl_joints = smpl_info['joints'][:,t,:,:]
   njoints = smpl_joints[:, mapping_ids]
   
   radius  = cfg.render_camera_radius
   n_samples = cfg.render_n_samples
   verts, colors, faces, bounds = layouts
   
   # colors = torch.tensor([[255,0,0] for i in range(len(verts[t]))])
   meshes = []
   mesh_colors = []

   for k in range(len(verts[t][:])):
      # if k == 0:
      #    continue
      mesh = make_mesh(verts[t][k], faces[t], colors[t][k][:3])
      meshes.append(mesh)
      mesh_colors.append(colors[t][k][:3])
   
   bb_min, bb_max = get_scene_bb(meshes)
   center = 0.5 * (bb_min + bb_max)
   center = [-center[0], -center[1], -center[2]]
   if len(out_flags) > 0 and not cfg.overwrite:
      return save_root, center, mesh_colors
   
   h, w = cfg.nerf_train_img_hw
   renderer = Renderer(w, h, alight=(255,255,255))
   flags = RenderFlags.RGBA

   
   # center = [0,0,0]

   joints_mesh = []
   for joints in njoints:
      joints = joints.cpu().numpy()
      for i in range(len(joints)):
         joints[i][1] = -joints[i][1]
         joints[i][2] = -joints[i][2]
         tfm = np.eye(4)
         tfm[:3, 3] = center
         joints_homo = np.ones((4,1))
         joints_homo[:3, 0] = joints[i]
         joints[i] = (tfm @ joints_homo)[:3, 0]
         
         
      sm = trimesh.creation.uv_sphere(radius=0.02)
      sm.visual.vertex_colors = [1.0, 0.0, 0.0]
      tfs = np.tile(np.eye(4), (len(joints), 1, 1))
      tfs[:,:3,3] = joints
      import pyrender
      m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
      joints_mesh.append(m)

   # cam_poses = spherical_camera_pose(n_samples, center, radius)
   cam_poses = spherical_camera_pose(n_samples, [0,0,0], radius)

   mesh_trans = Transform((center[0], center[1], center[2]))
   light_trans = Transform((10, 10, 10))
   for k,mesh in enumerate(meshes):
      renderer.add_mesh(mesh, mesh_trans)
      # renderer.scene.add_node(
         # renderer._gen_node("mesh", joints_mesh[k], 
         #                    tranlate=(center[0], center[1], center[2]),
         #                    rotation=(0,0,0,1)))
      # renderer.scene.add_node(
      #    pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh), 
      #                  )
      # )
      # renderer.scene.add_node(
      #    pyrender.Node(mesh=joints_mesh[k],
      #                  ))
      


   # for cam_pose in cam_poses: 
   #     renderer.add_camera(cam_pose, visiable=True)
   renderer.add_camera(cam_poses[0])

   # renderer.add_light('directlight', light_trans, np.ones(3), 10)

   if viewer:
      import pyrender
      pyrender.Viewer(renderer.scene)
      import pdb;pdb.set_trace()
      return 0
   # if len(out_flags) > 0 and not cfg.overwrite:
   #    return save_root, center, mesh_colors

   for i in range(n_samples):
      cam_node = renderer.camera_node
      renderer.scene.set_pose(cam_node, cam_poses[i])
      rgb, depth = renderer.render(flags)
      H, W, _ = rgb.shape
      rgb_copy = rgb.copy()
      for h in range(H):
         for w in range(W):
               if depth[h][w] == 0:
                  rgb_copy[h][w][3] = 0
      # import cv2
      # cv2.imshow('', rgb)
      # cv2.waitKey(0)         
      img = Image.fromarray(rgb_copy)
      img.save(os.path.join(img_save_path, "r_{}.png".format(i)), format='png')
      imageio.imsave(os.path.join(img_save_path, "r_{}_depth_{:0>4d}.png".format(i, i)), depth)

   def saveJson(filename, data):
      with open(filename, "w") as f:
         json.dump(data, f, indent=4)

   aspect = W / H
   xfov = renderer.camera.yfov * aspect
   trans_json = {
                  "camera_angle_x": xfov,
                  "camera_angle_y": renderer.camera.yfov,
                  "aabb_scale": cfg.aabb_scale,
                  "height": H,
                  "width": W,
                  "frames":[]
               }

   for i in range(n_samples):
      file_path = f'./{split}/r_{i}'
      trans_json['frames'].append(
         {
               "file_path": file_path,
               "transform_matrix":cam_poses[i].tolist()
         }
      )
   
   saveJson(f'{save_root}/transforms_{split}.json', trans_json)
   saveJson(f'{save_root}/transforms_test.json', trans_json)
   saveJson(f'{save_root}/transforms_val.json', trans_json)
   return save_root, center, mesh_colors
      

def gen_nerf_config_hashnerf(save_root, exp_name):
   txt_path = os.path.join(save_root, f"{exp_name}.txt")
   # if os.path.exists(txt_path):
   #    return txt_path
   basedir = os.path.join(save_root, 'logs')
   os.makedirs(basedir, exist_ok=True)
   with open(txt_path, "w") as f:
      f.write(f'expname = {exp_name}\n')
      f.write(f"basedir = {basedir}\n")
      f.write(f'datadir = {save_root}\n')
      f.write(f'dataset_type = blender\n')
      f.write('\n')
      f.write('no_batching = True\n')
      f.write('\n')
      f.write('use_viewdirs = True\n')
      f.write('white_bkgd = True\n')
      f.write('lrate_decay = 500\n')
      f.write('\n')
      f.write('N_samples = 64\n')
      f.write('N_importance = 128\n')
      f.write('N_rand = 1024\n')
      f.write('\n')
      f.write('precrop_iters = 500\n')
      f.write('precrop_frac = 0.5\n')
      f.write('\n')
      f.write('half_res = False\n')
   return txt_path


def gen_nerf_config_DVGO(save_root, exp_name):
   py_path = os.path.join(save_root, f"{exp_name}.py")
   if os.path.exists(py_path):
      return py_path
   basedir = os.path.join(save_root+'/', 'dvgo_logs')
   os.makedirs(basedir, exist_ok=True)
   with open(py_path, "w") as f:
      f.write('_base_ = \'D:/Python Code/CameraExtractor/filmingnerf/nerf/DirectVoxGO/configs/default.py\'\n')
      f.write('\n')
      f.write(f'expname = \'{exp_name}\'\n')
      f.write(f"basedir = \'{basedir}\'\n")
      f.write('\n')
      f.write(f'data = dict(\n')
      f.write(f'    datadir=\'{save_root}\',\n')
      f.write(f'    dataset_type=\'blender\',\n')
      f.write(f'    white_bkgd=True,\n')
      f.write(')')
   return py_path
