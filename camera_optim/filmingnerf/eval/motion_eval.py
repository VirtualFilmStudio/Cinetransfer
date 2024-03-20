import numpy as np
import torch


def evaluate_physics_metric(smpl_verts, smpl_joints, vis_mask):
    penetration_list = []
    floating_list = []
    skating_list = []

    vert_z = smpl_verts[..., 2]
    lowest_z = vert_z.min(dim=-1)[0]

    for i, mask in enumerate(vis_mask):
        # import pdb;pdb.set_trace()
        penetration = -lowest_z[i, mask==1].clamp_max(0) * 1000
        penetration_list += penetration.tolist()

    # compute floating
    for i, mask in enumerate(vis_mask):
        floating = lowest_z[i, mask==1].clamp_min(0) * 1000
        floating_list += floating.tolist()

    # compute foot sliding
    margin = 0.02
    for joints, mask in zip(smpl_joints, vis_mask):
        for t in range(len(mask)):
            if mask[t] == 1:
                cind = joints[t, :, 2].min(dim=-1)[1]
                if joints[t, cind, 2] <= margin and joints[t + 1, cind, 2] <= margin:
                    offset = joints[t + 1, cind, :2] - joints[t, cind, :2]
                    skate_i = torch.norm(offset).mean().item() * 1000
                else:
                    skate_i = 0.0
                skating_list.append(skate_i)

    print('penetration:', np.mean(penetration_list))
    print('floating:', np.mean(floating_list))
    print('skating:', np.mean(skating_list))
    