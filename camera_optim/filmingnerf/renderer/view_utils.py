import torch

import os
import cv2
import numpy as np
import torch
from PIL import Image

def filter_visible_meshes(verts, colors, faces, vis_mask=None, vis_opacity=False):
    """
    :param verts (B, T, V, 3)
    :param colors (B, 3)
    :param faces (F, 3)
    :param vis_mask (optional tensor, default None) (B, T) ternary mask
        -1 if not in frame
         0 if temporarily occluded
         1 if visible
    :param vis_opacity (optional bool, default False)
        if True, make occluded people alpha=0.5, otherwise alpha=1
    returns a list of T lists verts (Bi, V, 3), colors (Bi, 4), faces (F, 3)
    """
    B, T = verts.shape[:2]
    faces = [faces for t in range(T)]
    if vis_mask is None:
        verts = [verts[:, t] for t in range(T)]
        colors = [colors for t in range(T)]
        return verts, colors, faces

    # render occluded and visible, but not removed
    vis_mask = vis_mask >= 0
    if vis_opacity:
        alpha = 0.5 * (vis_mask[..., None] + 1)
    else:
        alpha = (vis_mask[..., None] >= 0).float()
    vert_list = [verts[vis_mask[:, t], t] for t in range(T)]
    colors = [
        torch.cat([colors[vis_mask[:, t]], alpha[vis_mask[:, t], t]], dim=-1)
        for t in range(T)
    ]
    bounds = get_bboxes(verts, vis_mask)
    return vert_list, colors, faces, bounds


def get_bboxes(verts, vis_mask):
    """
    return bb_min, bb_max, and mean for each track (B, 3) over entire trajectory
    :param verts (B, T, V, 3)
    :param vis_mask (B, T)
    """
    B, T, *_ = verts.shape
    bb_min, bb_max, mean = [], [], []
    for b in range(B):
        v = verts[b, vis_mask[b, :T]]  # (Tb, V, 3)
        bb_min.append(v.amin(dim=(0, 1)))
        bb_max.append(v.amax(dim=(0, 1)))
        mean.append(v.mean(dim=(0, 1)))
    bb_min = torch.stack(bb_min, dim=0)
    bb_max = torch.stack(bb_max, dim=0)
    mean = torch.stack(mean, dim=0)
    # point to a track that's long and close to the camera
    zs = mean[:, 2]
    counts = vis_mask[:, :T].sum(dim=-1)  # (B,)
    mask = counts < 0.8 * T
    zs[mask] = torch.inf
    sel = torch.argmin(zs)
    return bb_min.amin(dim=0), bb_max.amax(dim=0), mean[sel]

def track_to_colors(track_ids):
    """
    :param track_ids (B)
    """
    color_map = torch.from_numpy(get_colors()).to(track_ids)
    return color_map[track_ids] / 255  # (B, 3)

def get_colors():
    color_file = os.path.abspath(os.path.join(__file__, "../colors.txt"))
    RGB_tuples = np.vstack(
        [
            np.loadtxt(color_file, skiprows=0),
            np.random.uniform(0, 255, size=(10000, 3)),
            [[0, 0, 0]],
        ]
    )
    b = np.where(RGB_tuples == 0)
    RGB_tuples[b] = 1
    return RGB_tuples.astype(np.float32)

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)