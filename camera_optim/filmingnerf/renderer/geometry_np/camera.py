import numpy as np
import math

def intrasphere_sampling(n_samples, r):
    angle1 = np.random.rand(1, n_samples)*2*np.pi
    angle2 = math.acos(rand(1, n_samples)*2-1)
    pass

def spherical_sample(n_sample):
    u, v = np.random.rand(2, n_sample)
    # u = np.random.uniform(0, 1)
    # v = np.random.uniform(0, 1)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    X = np.sin(phi) * np.cos(theta)
    Y = np.sin(phi) * np.sin(theta)
    Z = np.cos(phi)
    out = []
    for x, y, z in zip(X, Y, Z):
        out.append([x, y, z])
    return out

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-1]
    R = R.reshape(-1, 3, 3)
    t = np.expand_dims(t, axis=2)
    T = np.concatenate((R, t), axis=2)
    print(dims)
    bottom = np.repeat((
        np.array([0, 0, 0, 1])
        .reshape(*(1,) * len(dims), 1, 4)
    ), dims, axis=0)
    
    T = np.concatenate((T, bottom), axis=1)
    return T

def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def camera_look_at(eye, target, up):
    up = normalize(up)
    z = normalize(eye - target)
    x = normalize(np.cross(up, z))
    y = normalize(np.cross(z, x))
    mat = np.eye(4)

    mat[:3,:3] = np.array([x,y,z]).T
    mat[:3, 3] = eye
    return mat