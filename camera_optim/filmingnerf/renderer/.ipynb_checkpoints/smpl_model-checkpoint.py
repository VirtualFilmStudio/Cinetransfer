import numpy as np
import torch
import os
import trimesh

from smplx import SMPL, SMPLH, SMPLX
from smplx.vertex_ids import vertex_ids
from smplx.joint_names import SMPLH_JOINT_NAMES
from smplx.utils import Struct

from .specs import SMPL_JOINTS

class SmplModel(object):

    def __init__(
        self,
        bm_path,
        num_betas=16,
        batch_size=1,
        num_expressions=10,
        use_vtx_selector=True,
        model_type="smplh",
        kid_template_path=None,
    ):
        super(SmplModel, self).__init__()
        """
        Creates the body model object at the given path.

        :param bm_path: path to the body model pkl file
        :param num_expressions: only for smplx
        :param model_type: one of [smpl, smplh, smplx]
        :param use_vtx_selector: if true, returns additional vertices as joints that correspond to OpenPose joints
        """
        self.use_vtx_selector = use_vtx_selector
        cur_vertex_ids = None
        
        if self.use_vtx_selector:
            cur_vertex_ids = vertex_ids[model_type]
   
        data_struct = None
        if ".npz" in bm_path:
            # smplx does not support .npz by default, so have to load in manually
            smpl_dict = np.load(bm_path, encoding="latin1")
            data_struct = Struct(**smpl_dict)
            # print(smpl_dict.files)
            if model_type == "smplh":
                data_struct.hands_componentsl = np.zeros((0))
                data_struct.hands_componentsr = np.zeros((0))
                data_struct.hands_meanl = np.zeros((15 * 3))
                data_struct.hands_meanr = np.zeros((15 * 3))
                V, D, B = data_struct.shapedirs.shape
                data_struct.shapedirs = np.concatenate(
                    [data_struct.shapedirs, np.zeros((V, D, SMPL.SHAPE_SPACE_DIM - B))],
                    axis=-1,
                )  # super hacky way to let smplh use 16-size beta
        kwargs = {
            "model_type": model_type,
            "data_struct": data_struct,
            "num_betas": num_betas,
            "batch_size": batch_size,
            "num_expression_coeffs": num_expressions,
            "vertex_ids": cur_vertex_ids,
            "use_pca": False,
            "flat_hand_mean": False,
        }
        if kid_template_path is not None:
            kwargs["kid_template_path"] = kid_template_path
            kwargs["age"] = "kid"

        assert model_type in ["smpl", "smplh", "smplx"]
        if model_type == "smpl":
            self.bm = SMPL(bm_path, **kwargs)
            self.num_joints = SMPL.NUM_JOINTS
        elif model_type == "smplh":
            self.bm = SMPLH(bm_path, **kwargs)
            self.num_joints = SMPLH.NUM_JOINTS
        elif model_type == "smplx":
            self.bm = SMPLX(bm_path, **kwargs)
            self.num_joints = SMPLX.NUM_JOINTS

        self.model_type = model_type

    def load_param(
        self,
        root_orient=None,
        pose_body=None,
        pose_hand=None,
        pose_jaw=None,
        pose_eye=None,
        betas=None,
        trans=None,
        dmpls=None,
        expression=None,
        return_dict=False,
        **kwargs):
        assert dmpls is None
        out_obj = self.bm(
            betas=betas,
            global_orient=root_orient,
            body_pose=pose_body,
            left_hand_pose=None
            if pose_hand is None
            else pose_hand[:, : (SMPLH.NUM_HAND_JOINTS * 3)],
            right_hand_pose=None
            if pose_hand is None
            else pose_hand[:, (SMPLH.NUM_HAND_JOINTS * 3) :],
            transl=trans,
            expression=expression,
            jaw_pose=pose_jaw,
            leye_pose=None if pose_eye is None else pose_eye[:, :3],
            reye_pose=None if pose_eye is None else pose_eye[:, 3:],
            return_full_pose=True,
            **kwargs
        )
        
        out = {
            "v": out_obj.vertices,
            "f": self.bm.faces_tensor,
            "betas": out_obj.betas,
            "Jtr": out_obj.joints,
            "pose_body": out_obj.body_pose,
            "full_pose": out_obj.full_pose,
        }
        if self.model_type in ["smplh", "smplx"]:
            out["pose_hand"] = torch.cat(
                [out_obj.left_hand_pose, out_obj.right_hand_pose], dim=-1
            )
        if self.model_type == "smplx":
            out["pose_jaw"] = out_obj.jaw_pose
            out["pose_eye"] = pose_eye

        if not self.use_vtx_selector:
            # don't need extra joints
            out["Jtr"] = out["Jtr"][:, : self.num_joints + 1]  # add one for the root

        if not return_dict:
            out = Struct(**out)
 
        return out
    
def run_smpl(body_model, trans, root_orient, body_pose, betas=None):
    """
    Forward pass of the SMPL model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : (optional) B x D
    """
    B, T, _ = trans.shape
    bm_batch_size = body_model.bm.batch_size
    print(B, T, bm_batch_size)
    assert bm_batch_size % B == 0
    seq_len = bm_batch_size // B
    bm_num_betas = body_model.bm.num_betas
    J_BODY = len(SMPL_JOINTS) - 1  # all joints except root

    if T == 1:
        # must expand to use with body model
        trans = trans.expand(B, seq_len, 3)
        root_orient = root_orient.expand(B, seq_len, 3)
        body_pose = body_pose.expand(B, seq_len, J_BODY * 3)
    elif T != seq_len:
        trans, root_orient, body_pose = zero_pad_tensors(
            [trans, root_orient, body_pose], seq_len - T
        )
    if betas is None:
        betas = torch.zeros(B, bm_num_betas, device=trans.device)
    betas = betas.reshape((B, 1, bm_num_betas)).expand((B, seq_len, bm_num_betas))
    
    smpl_body = body_model.load_param(
        pose_body=body_pose.reshape((B * seq_len, -1)),
        pose_hand=None,
        betas=betas.reshape((B * seq_len, -1)),
        root_orient=root_orient.reshape((B * seq_len, -1)),
        trans=trans.reshape((B * seq_len, -1)),
    )
    joints_names = {}
    for i in range(len(SMPLH_JOINT_NAMES)):
        joints_names.update({
            i:SMPLH_JOINT_NAMES[i]
        })
    mapping_ids = [52,12,17,19,21,16,18,20,0,2,5,8,1,4,7,53,54,55,56,57,58,59,60,61,62]
    # for i in mapping_ids:
    #     print(joints_names[i])
    # import pdb;pdb.set_trace()
    return {
        "joints_names": joints_names,
        "mapping_ids": mapping_ids,
        "joints": smpl_body.Jtr.reshape(B, seq_len, -1, 3)[:, :T],
        "vertices": smpl_body.v.reshape(B, seq_len, -1, 3)[:, :T],
        "faces": smpl_body.f
    }

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
    return color_map[track_ids.long()] # (B, 3)


def get_colors():
    #     color_file = os.path.abspath(os.path.join(__file__, "../colors_phalp.txt"))
    color_file = os.path.abspath(os.path.join(__file__, "../colors.txt"))
    RGB_tuples = np.vstack(
        [
            np.loadtxt(color_file, skiprows=0),
            #             np.loadtxt(color_file, skiprows=1),
            np.random.uniform(0, 255, size=(10000, 3)),
            [[0, 0, 0]],
        ]
    )
    b = np.where(RGB_tuples == 0)
    RGB_tuples[b] = 1
    return RGB_tuples.astype(np.float32)

def zero_pad_tensors(pad_list, pad_size):
    """
    Assumes tensors in pad_list are B x T x D and pad temporal dimension
    """
    B = pad_list[0].size(0)
    new_pad_list = []
    for pad_idx, pad_tensor in enumerate(pad_list):
        padding = torch.zeros((B, pad_size, pad_tensor.size(2))).to(pad_tensor)
        new_pad_list.append(torch.cat([pad_tensor, padding], dim=1))
    return new_pad_list

