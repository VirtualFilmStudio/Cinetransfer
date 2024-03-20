import torch
import os
import numpy as np
import torch.nn as nn

from ..renderer.geometry.camera import *

def vec2ss_matrix(vector):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix


class BaseCameraModel(nn.Module):
    def __init__(self, xfov=None, yfov=None):
        super(BaseCameraModel, self).__init__()
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.theta = nn.Parameter(torch.normal(0., 1e-6, size=()))
        self.xfov = xfov
        self.yfov = yfov
        self.R = None
        self.t = None

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
        # import pdb;pdb.set_trace() 
        Pc = Pc / Pc[2,0]
        # Pc[0,0] = Pc[0,0]/Pc[2,0]
        # Pc[1,0] = Pc[1,0]/Pc[2,0]
        # Pc[2,0] = Pc[2,0]/Pc[2,0]
        
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
        # import pdb;pdb.set_trace()
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

    def forward(self, x):
        return self.transform(x)
    

class DoubleTajCameraModel(nn.Module):
    def __init__(self, look_at, cam_pos):
        super(DoubleTajCameraModel, self).__init__()
        self.lookAt = nn.Parameter(look_at)
        self.camPos = nn.Parameter(cam_pos)

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
        # import pdb;pdb.set_trace() 
        Pc = Pc / Pc[2,0]
        # Pc[0,0] = Pc[0,0]/Pc[2,0]
        # Pc[1,0] = Pc[1,0]/Pc[2,0]
        # Pc[2,0] = Pc[2,0]/Pc[2,0]
        
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

    def forward(self):
        camera_pose = lookat_matrix_ngp(self.camPos, self.lookAt, torch.Tensor([0, -1, 0]))
        self.T = lookat_matrix(self.camPos, self.lookAt, torch.Tensor([0, 1, 0]))
        return self.nerf_matrix_to_ngp(camera_pose)
    
    def extract_params(self):
        look_at = self.lookAt.tolist()
        cam_pos = self.camPos.tolist()
        return look_at, cam_pos
    

class IntrinsicsCameraModel(BaseCameraModel):
    def __init__(self):
        super(IntrinsicsCameraModel, self).__init__()
        self.focal = nn.Parameter(torch.normal(0., 1e-6, size=()))
        
    def forward(self, x, H, W):
        K = torch.tensor([
            [self.focal, 0, 0.5*W],
            [0, self.focal, 0.5*H],
            [0, 0, 1]
        ])
        return self.transform(x), K

class SphericalCameraModel(nn.Module):
    def __init__(self, params):
       super(SphericalCameraModel, self).__init__()
       f_pos, theta, phi, scale = params
       self.w = nn.Parameter(f_pos)
       self.theta = nn.Parameter(theta)
       self.phi = nn.Parameter(phi)
       self.a = nn.Parameter(scale)

    def extract_params(self):
        x = self.w[0].item()
        y = self.w[1].item()
        z = self.w[2].item()
        scale = self.a.item()
        theta = self.theta.item()
        phi = self.phi.item()
        return [x,y,z,scale,phi,theta]

    def reproject(self, P, offset, hwf):
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

    def forward(self):
        up = torch.Tensor([0, -1, 0])
        cam_x = self.a * torch.sin(self.phi) * torch.cos(self.theta)
        cam_y = self.a * torch.sin(self.phi) * torch.sin(self.theta)
        cam_z = self.a * torch.cos(self.phi)
        camera_pos = torch.Tensor([cam_x, cam_y, cam_z])
        camera_pose = lookat_matrix_ngp(camera_pos, self.w, up)
        self.T = lookat_matrix(camera_pos, self.w, torch.Tensor([0, 1, 0]))
        return self.nerf_matrix_to_ngp(camera_pose)

from ..renderer.geometry.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix
class SixDimCameraModel(nn.Module):
    def __init__(self, R, T):
        super(SixDimCameraModel, self).__init__()
        R_6d_init = matrix_to_rotation_6d(R)
        self.R_6d = nn.Parameter(R_6d_init.clone())
        self.T = nn.Parameter(T.clone())

    def forward(self, t):
        R = rotation_6d_to_matrix(self.R_6d)[t]
        t = self.T[t]
        exp_i = torch.zeros((4, 4))
        exp_i[:3, :3] = R
        exp_i[:3, 3] = t
        exp_i[3, 3] = 1.
        return exp_i
        
        