import torch

from .view_utils import track_to_colors, filter_visible_meshes 

def smpl_to_geometry(verts, faces, vis_mask=None, track_ids=None):
    """
    :param verts (B, T, V, 3)
    :param faces (F, 3)
    :param vis_mask (optional) (B, T) visibility of each person
    :param track_ids (optional) (B,)
    returns list of T verts (B, V, 3), faces (F, 3), colors (B, 3)
    where B is different depending on the visibility of the people
    """
    B, T = verts.shape[:2]
    device = verts.device

    # (B, 3)
    colors = (
        track_to_colors(track_ids)
        if track_ids is not None
        else torch.ones(B, 3, device) * 0.5
    )

    # list T (B, V, 3), T (B, 3), T (F, 3)
    return filter_visible_meshes(verts, colors, faces, vis_mask)