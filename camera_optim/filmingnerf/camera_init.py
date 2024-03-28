import torch
import random
import numpy as np

from .renderer.geometry_np.camera import spherical_sample, camera_look_at
from .renderer.geometry.rotation import *
from .renderer.geometry.camera import *

def spherical_sampling(n_samples, center, radius, offset=np.array([0,0,0])):
    sample_pos = spherical_sample(n_samples)
    up = np.array([0,0,1])
    camera_poses = []
    for pos in sample_pos:
        src_pos = [i*radius for i in pos]
        target = np.array(center) + offset
        camera_pose = camera_look_at(src_pos, target, up)
        camera_poses.append(camera_pose)
    return camera_poses


def double_spherical_sampling(cam_samples, r_cam, tar_samples, r_tar):
    sample_pos = spherical_sample(cam_samples)
    up = np.array([0,1,0])
    camera_poses = []
    for pos in sample_pos:
        src_pos = [i*r_cam for i in pos]
        targets = spherical_sample(tar_samples)
        for target in targets:
            target = np.array([i*r_tar for i in target])
            camera_pose = camera_look_at(src_pos, target, up)
            camera_poses.append(camera_pose)
    return camera_poses


def double_cinematic_sampling(device, delta, up=torch.Tensor([0,1,0]), forward=torch.Tensor([1,0,0]), samples=[1,1,1,1]):
    up = up.to(device)
    forward = forward.to(device)
    delta = torch.tensor(delta)
    cinecore_rot = torch.Tensor([0,0,0,0])
    targets = spherical_sample(samples[0])
    camera_poses = []
    for cinecore_pos in targets:
        cinecore_pos = torch.mul(torch.Tensor(cinecore_pos), 0.3)
        for i in np.arange(40,70,(70-50)/(samples[1]+2))[1:-1]:
            for j in np.arange(-180, 180, 360/(samples[2]+2))[1:-1]: 
                for k in np.arange(-180, 180, 360/(samples[3]+2))[1:-1]:
                    gamma = torch.tensor(0.5)
                    Alpha = torch.tensor(i).long()
                    Phi = torch.tensor(j).long()
                    Theta = torch.tensor(k).long()
                    cinematic_params = [cinecore_pos, cinecore_rot, gamma, delta, Alpha, Phi, Theta]
                    pose = CalCinematicSpace(cinematic_params, up, forward)
                    camera_poses.append(pose)
    return camera_poses


def cinematic_sampling(device, delta, up=torch.Tensor([0,1,0]), forward=torch.Tensor([1,0,0]), samples=[1,1,1,1]):
    up = up.to(device)
    forward = forward.to(device)
    delta = torch.tensor(delta)
    cinecore_pos = torch.Tensor([0,0,0])
    cinecore_rot = torch.Tensor([0,0,0,0])
    camera_poses = []
    for m in np.arange(0, 1, 1 / (samples[0]+2))[1:-1]:
        for i in np.arange(50,70,(70-50)/(samples[1]+2))[1:-1]:
            for j in np.arange(-180, 180, 360/(samples[2]+2))[1:-1]: 
                for k in np.arange(-180, 180, 360/(samples[3]+2))[1:-1]:
                    gamma = torch.tensor(m)
                    Alpha = torch.tensor(70).long()
                    Phi = torch.tensor(j).long()
                    Theta = torch.tensor(k).long()
                    cinematic_params = [cinecore_pos, cinecore_rot, gamma, delta, Alpha, Phi, Theta]
                    pose = CalCinematicSpace(cinematic_params, up, forward)
                    camera_poses.append(pose)
    return camera_poses

def nerf_matrix_to_ngp(pose, scale=0.8, offset=[0, 0, 0]):
        # for the fox dataset, 0.33 scales camera radius to ~ 2
        new_pose = torch.Tensor([
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
            [0, 0, 0, 1],
        ])
        return new_pose

def CalSphericalSpace(params, up):
    f_pos, theta, phi, scale = params
    cam_x = scale * torch.sin(phi) * torch.cos(theta)
    cam_y = scale * torch.sin(phi) * torch.sin(theta)
    cam_z = scale * torch.cos(phi)
    camera_pos = torch.Tensor([cam_x, cam_y, cam_z])
    camera_pose = lookat_matrix(camera_pos, f_pos, up)
    return camera_pose, camera_pos

def CalSphericalSpaceNGP(params, up):
    f_pos, theta, phi, scale = params
    cam_x = scale * torch.sin(phi) * torch.cos(theta)
    cam_y = scale * torch.sin(phi) * torch.sin(theta)
    cam_z = scale * torch.cos(phi)
    camera_pos = torch.Tensor([cam_x, cam_y, cam_z])
    camera_pose = lookat_matrix_ngp(camera_pos, f_pos, up)
    return nerf_matrix_to_ngp(camera_pose)


def CalCinematicSpace(params, up, forward):
    cinecore_pos, cinecore_rot, gamma, delta, Alpha, Phi, Theta = params
    cine_rot_m = quaternion_to_rotation_matrix(cinecore_rot.unsqueeze(0)).squeeze(0)
    Q_z = cine_rot_m@forward.unsqueeze(1).squeeze(1)

    D_AQ = gamma * delta
    D_QB = (1 - gamma) * delta
    AQ = Q_z * (-D_AQ)
    QB = Q_z * D_QB

    T_A = cinecore_pos + AQ
    T_B = cinecore_pos + QB
    T_C = cinecore_pos

    AB = T_B - T_A
    alpha = Alpha*torch.pi/180
    phi = Phi*torch.pi/180
    theta = Theta*torch.pi/180

    IO0 = F.normalize(torch.cross(up, AB), dim=0) * (delta / (2 * torch.tan(alpha)))
    angle_axis = phi.unsqueeze(-1) * F.normalize(AB,dim=0)
    q_phi = angle_axis_to_quaternion(angle_axis)

    IO = quaternion_to_rotation_matrix(q_phi.unsqueeze(0)).squeeze(0)@IO0

    UP = torch.cross(AB, IO)

    if Alpha > 90:
        UP = -UP

    AO = 0.5 * AB + IO
    O_pos = T_A + AO

    angle_axis = theta.unsqueeze(-1) * F.normalize(UP,dim=0)
    q_theta = angle_axis_to_quaternion(angle_axis)

    OB = T_B - O_pos
    P = quaternion_to_rotation_matrix(q_theta.unsqueeze(0)).squeeze(0)@OB + O_pos

    camera_pos = P

    pose = lookat_matrix(camera_pos.unsqueeze(0), cinecore_pos.unsqueeze(0), up)
    return pose[0]