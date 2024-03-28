import torch
import os
import numpy as np
import torch.nn as nn
from ..renderer.geometry.camera import *
from ..renderer.geometry.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix

def gen_mlp(in_channel, out_channel, inner_size=512, is_linear=False):
    if(is_linear==False):
        mlp = nn.Sequential(
        nn.Linear(in_channel, inner_size),
        nn.ReLU(),
        nn.Linear(inner_size, out_channel)
        )
    else:
        mlp = nn.Sequential(
        nn.Linear(in_channel, inner_size),
        nn.Linear(inner_size, out_channel)
        )
    return mlp


class CameraSequencer(nn.Module):
    def __init__(self, frame_num):
        super(CameraSequencer, self).__init__()
        self.F = torch.zeros((frame_num, 6))
        self.W_x = gen_mlp(1, 1)
        self.W_y = gen_mlp(1, 1)
        self.W_z = gen_mlp(1, 1)
        self.G_theta = gen_mlp(1, 1)
        self.G_phi = gen_mlp(1, 1)
        self.V = gen_mlp(1, 1)

    def set_F0(self, f0):
        self.F[0] = f0

    def reproject(self, P, offset, hwf, t):
        # P[0] = -P[0]
        P[1] = -P[1]
        P[2] = -P[2]
        
        tfm = torch.eye(4)
        tfm[:3, 3] = offset
        P_homo = torch.ones((4,1))
        P_homo[:3, 0] = P
        P = (tfm @ P_homo)[:3, 0] 

        Pw = torch.ones((4,1))
        Pw[:3,:] = P.unsqueeze(0).T
        
        h, w, fx, fy = hwf
        K = torch.tensor([
            [fx, 0, 0.5*w],
            [0, fy, 0.5*h],
            [0, 0, 1]
        ])

        Pc = (torch.inverse(self.cal_camera_pose(t)) @ Pw)[:3]

        Pc = Pc / Pc[2,0]

        po = (K @ Pc.double()).T.squeeze(0)
        
        po = po * torch.Tensor([-1,1,1])
        po = po + torch.Tensor([w, 0, 0])
            
        return po
    
    def nerf_matrix_to_ngp(self, pose, scale=0.8, offset=[0, 0, 0]):
        # for the fox dataset, 0.33 scales camera radius to ~ 2
        new_pose = torch.Tensor([
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
            [0, 0, 0, 1],
        ])
        return new_pose
    
    def cal_camera_pose(self, t):
        up = torch.Tensor([0, 1, 0])
        x, y, z, theta, phi, scale = self.F[t]
        cam_x = scale * torch.sin(phi) * torch.cos(theta)
        cam_y = scale * torch.sin(phi) * torch.sin(theta)
        cam_z = scale * torch.cos(phi)
        camera_pos = torch.Tensor([cam_x, cam_y, cam_z])
        target_pos = torch.Tensor([x, y, z])
        camera_pose = lookat_matrix(camera_pos, target_pos, up)
        return camera_pose
    
    def cal_camera_pose_ngp(self, t):
        up = torch.Tensor([0, -1, 0])
        x, y, z, theta, phi, scale = self.F[t]
        cam_x = scale * torch.sin(phi) * torch.cos(theta)
        cam_y = scale * torch.sin(phi) * torch.sin(theta)
        cam_z = scale * torch.cos(phi)
        camera_pos = torch.Tensor([cam_x, cam_y, cam_z])
        target_pos = torch.Tensor([x, y, z])
        camera_pose = lookat_matrix_ngp(camera_pos, target_pos, up)
        return self.nerf_matrix_to_ngp(camera_pose)
    
    def cal_params(self, t):
        p = self.F[t]
        t_in = torch.tensor(t).float().unsqueeze(0)
        x = self.W_x(t_in) + p[0]
        y = self.W_y(t_in) + p[1]
        z = self.W_z(t_in) + p[2]
        theta = self.G_theta(t_in) + p[3]
        phi = self.G_phi(t_in) + p[4]
        scale = self.V(t_in) + p[5]
        return [x, y, z, theta, phi, scale]

    def forward(self, t):
        self.F[t] = torch.stack(self.cal_params(t), dim=1).squeeze(0)
        return self.cal_camera_pose_ngp(t)
    
    def show_fig(self, T, save=False, save_path=None):
        import matplotlib.pyplot as plt
        look_at_points = []
        cam_pos_points = []
        for t in range(0, T):
            t_in = torch.tensor(t).float().unsqueeze(0)
            x, y, z, theta, phi, scale = self.cal_params(t)
            cam_x = scale * torch.sin(phi) * torch.cos(theta)
            cam_y = scale * torch.sin(phi) * torch.sin(theta)
            cam_z = scale * torch.cos(phi)
            look_at_points.append([x.item(), y.item(), z.item()])
            cam_pos_points.append([cam_x.item(),cam_y.item(),cam_z.item()])

        look_at_points = np.array(look_at_points)
        cam_pos_points = np.array(cam_pos_points)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for m, points, label in [('o', look_at_points, 'lookAt'), ('^', cam_pos_points, 'camPos')]:
            xs = points[:,0]
            ys = points[:,1]
            zs = points[:,2]
            ax.scatter(xs, ys, zs, marker=m, label=label)
        if save:
            plt.savefig(save_path)
        else:
            plt.show()

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight,0,0.001)
        nn.init.constant_(m.bias, 0)

def vec2ss_matrix(vector):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix

class CameraSequencerBase(nn.Module):
    def __init__(self):
        super(CameraSequencerBase, self).__init__()
        self.init_w = torch.normal(0., 1e-6, size=(3,))
        self.init_v = torch.normal(0., 1e-6, size=(3,))
        # self.W = gen_mlp(1, 6, inner_size=512)
        self.W = gen_mlp(1, 3, inner_size=512)
        self.V = gen_mlp(1, 3, inner_size=512)
        self.theta = nn.Parameter(torch.normal(0., 1e-6, size=()))

    def reproject(self, P, offset, hwf):
        # P[0] = -P[0]
        P[1] = -P[1]
        P[2] = -P[2]
        
        tfm = torch.eye(4)
        tfm[:3, 3] = offset
        P_homo = torch.ones((4,1))
        P_homo[:3, 0] = P
        P = (tfm @ P_homo)[:3, 0] 

        Pw = torch.ones((4,1))
        Pw[:3,:] = P.unsqueeze(0).T
        
        h, w, fx, fy = hwf
        K = torch.tensor([
            [fx, 0, 0.5*w],
            [0, fy, 0.5*h],
            [0, 0, 1]
        ])

        Pc = (torch.inverse(self.T) @ Pw)[:3]
        Pc = Pc / Pc[2,0]
        
        po = (K @ Pc.double()).T.squeeze(0)
        
        po = po * torch.Tensor([-1,1,1])
        po = po + torch.Tensor([w, 0, 0])  
        return po

    def transform(self, x):
        exp_i = torch.zeros((4, 4))
        w_skewsym = vec2ss_matrix(self.w)
        exp_i[:3, :3] = torch.eye(3) + torch.sin(self.theta) * w_skewsym + (1 - torch.cos(self.theta)) * torch.matmul(w_skewsym, w_skewsym)
        exp_i[:3, 3] = torch.matmul(torch.eye(3) * self.theta + (1 - torch.cos(self.theta)) * w_skewsym + (self.theta - torch.sin(self.theta)) * torch.matmul(w_skewsym, w_skewsym), self.v)
        exp_i[3, 3] = 1.
        T_i = torch.matmul(exp_i, x)
        self.T = T_i
        return T_i
    
    def nerf_matrix_to_ngp(self, pose, scale=0.8, offset=[0, 0, 0]):
        # for the fox dataset, 0.33 scales camera radius to ~ 2
        new_pose = torch.Tensor([
            [pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3] * scale + offset[0]],
            [pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3] * scale + offset[1]],
            [pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3] * scale + offset[2]],
            [0, 0, 0, 1],
        ])
        return new_pose
    
    def ngp_to_nerf_matrix(self, pose, scale=0.8, offset=[0, 0, 0]):
        # for the fox dataset, 0.33 scales camera radius to ~ 2
        new_pose = torch.Tensor([
            [pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3] / scale - offset[0]],
            [pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3] / scale - offset[1]],
            [pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3] / scale - offset[2]],
            [0, 0, 0, 1],
        ])
        return new_pose

    def forward(self, x, t):
        t_in = torch.tensor(t).float().unsqueeze(0)
        self.w = self.W(t_in)
        self.v = self.V(t_in)
        return self.transform(x)
    
    def show_fig(self, T, save=False, save_path=None):
        import matplotlib.pyplot as plt
        X = torch.arange(0, T).float().unsqueeze(1)
        X = X * (1/(T-1))

        pos_points = np.array(self.V(X).tolist())
        rot_points = np.array(self.W(X).tolist())

        fig = plt.figure()
        ax = fig.add_subplot(121,projection='3d')
        bx = fig.add_subplot(122,projection='3d')

        xs = pos_points[:,0]
        ys = pos_points[:,1]
        zs = pos_points[:,2]
        ax.scatter(xs, ys, zs, marker='o')

        xs = rot_points[:,0]
        ys = rot_points[:,1]
        zs = rot_points[:,2]
        bx.scatter(xs, ys, zs, marker='^')

        if save:
            plt.savefig(save_path)
        else:
            plt.show()

class CameraSequencerTwo(nn.Module):
    def __init__(self, look_at, cam_pos):
        super(CameraSequencerTwo, self).__init__()
        self.lookAt = look_at
        self.camPos = cam_pos

        self.W = gen_mlp(1, 3, inner_size=512)
        self.V = gen_mlp(1, 3, inner_size=512)

    def reproject(self, P, offset, hwf, t):
        # P[0] = -P[0]
        P[1] = -P[1]
        P[2] = -P[2]
        
        tfm = torch.eye(4)
        tfm[:3, 3] = offset
        P_homo = torch.ones((4,1))
        P_homo[:3, 0] = P
        P = (tfm @ P_homo)[:3, 0] 

        Pw = torch.ones((4,1))
        Pw[:3,:] = P.unsqueeze(0).T
        
        h, w, fx, fy = hwf
        K = torch.tensor([
            [fx, 0, 0.5*w],
            [0, fy, 0.5*h],
            [0, 0, 1]
        ])

        Pc = (torch.inverse(self.T) @ Pw)[:3]

        Pc = Pc / Pc[2,0]

        po = (K @ Pc.double()).T.squeeze(0)
        
        po = po * torch.Tensor([-1,1,1])
        po = po + torch.Tensor([w, 0, 0])
            
        return po
    
    def nerf_matrix_to_ngp(self, pose, scale=0.8, offset=[0, 0, 0]):
        # for the fox dataset, 0.33 scales camera radius to ~ 2
        new_pose = torch.Tensor([
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
            [0, 0, 0, 1],
        ])
        return new_pose
    
    def forward(self, t):
        t_in = torch.tensor(t).float().unsqueeze(0)
        
        new_look_at = self.W(t_in) + self.lookAt
        new_cam_pos = self.V(t_in) + self.camPos
  
        camera_pose = lookat_matrix_ngp(new_cam_pos, new_look_at, torch.Tensor([0, -1, 0]))
        self.T = lookat_matrix(new_cam_pos, new_look_at, torch.Tensor([0, 1, 0]))

        return self.nerf_matrix_to_ngp(camera_pose)
    
    def show_fig(self, T, save=False, save_path=None):
        import matplotlib.pyplot as plt
        X = torch.arange(0, T).float().unsqueeze(1)
        X = X * (1/(T-1))

        look_at_points = np.array(self.W(X).tolist())
        cam_pos_points = np.array(self.V(X).tolist())

        fig = plt.figure()
        ax = fig.add_subplot(121,projection='3d')
        bx = fig.add_subplot(122,projection='3d')

        xs = look_at_points[:,0]
        ys = look_at_points[:,1]
        zs = look_at_points[:,2]
        ax.scatter(xs, ys, zs, marker='o')

        xs = cam_pos_points[:,0]
        ys = cam_pos_points[:,1]
        zs = cam_pos_points[:,2]
        bx.scatter(xs, ys, zs, marker='^')

        if save:
            plt.savefig(save_path)
        else:
            plt.show()

    