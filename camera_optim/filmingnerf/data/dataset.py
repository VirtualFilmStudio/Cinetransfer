import os
import numpy as np
import json
import imageio
import cv2

import torch
from torch.utils.data import Dataset

from .tools import read_keypoints, read_image

MAX_NUM_TRACKS = 12
MIN_TRACK_LEN = 20
MIN_KEYP_CONF = 0.4


def get_data_from_cfg(cfg):
    data_source = {
        "images": os.path.join(cfg.data_root, cfg.seq_name,'images'),
        "nerf": os.path.join(cfg.data_root, cfg.seq_name, 'nerf'),
        "shots": os.path.join(cfg.data_root, cfg.seq_name, 'shot_dics',  f"{cfg.seq_name}.json"),
        "masks": os.path.join(cfg.data_root, cfg.seq_name, 'masks'),
        "tracks": os.path.join(cfg.data_root, cfg.seq_name, "track_preds", ),
        "cameras": os.path.join(cfg.data_root, cfg.seq_name,  "cameras", 'shot-0'),
        "mask_root": os.path.join(cfg.data_root, cfg.seq_name, "phalp_out")    
    }
    return CameraTrajectoryDataset(data_source, cfg.seq_name, cfg.start_id, cfg.end_id, cfg.nerf_render_img_hw)

class CameraTrajectoryDataset(Dataset):
    def __init__(
        self,
        data_sources,
        seq_name,
        start_idx,
        end_idx,
        resize_hw=None
    ):
        self.seq_name = seq_name,
        self.data_sources = data_sources
        self.resize_hw = resize_hw

        # load images path
        self.rgb_imgs_path = self.load_images_path(0, start_idx, end_idx)
        self.mask_imgs_path  = self.load_images_path(0, start_idx, end_idx, 'mask')
        self.load_track_data()

        self.sel_img_names = self.img_names[self.start_idx:self.end_idx]

        # used to cache data
        self.data_dict = {}
        self.load_data()
        
    def load_images_path(self, shot_idx, start_idx, end_idx, img_type='rgb'):
        shots_path = self.data_sources["shots"]
    
        assert os.path.isfile(shots_path)
        with open(shots_path, "r") as f:
            shots_dict = json.load(f)
        img_names = sorted(shots_dict.keys())
        shot_mask = np.array([shots_dict[x] == shot_idx for x in img_names])
        idcs = np.where(shot_mask)[0]
        img_files = [img_names[i] for i in idcs]
        print(f"FOUND {len(idcs)}/{len(shots_dict)} FRAMES ({img_type}) FOR SHOT {shot_idx}")

        if img_type == 'rgb':
            img_dir = self.data_sources["images"]
        elif img_type == 'mask':
            img_dir = self.data_sources["masks"]

        end_idx = end_idx if end_idx > 0 else len(img_files)
        self.data_start, self.data_end = start_idx, end_idx
        img_files = img_files[start_idx:end_idx]
        self.img_names = [
            os.path.splitext(os.path.basename(f))[0] for f in img_files]
        num_imgs = len(self.img_names)
        
        print(img_dir)
        assert os.path.isdir(img_dir)
        img_paths = [os.path.join(img_dir, f) for f in img_files]
        img_h, img_w = imageio.imread(img_paths[0]).shape[:2]
        self.img_size = img_w, img_h
        print(f"USING TOTAL {num_imgs} {img_w}x{img_h}  {img_type} IMGS")
        return img_paths

    def load_track_data(self, tid_spec='all'):
        track_root = self.data_sources["tracks"]
        if tid_spec == "all" or tid_spec.startswith("longest"):
            n_tracks = MAX_NUM_TRACKS
            if tid_spec.startswith("longest"):
                n_tracks = int(tid_spec.split("-")[1])
            # get the longest tracks in the selected shot
            track_ids = sorted(os.listdir(track_root))
            track_paths = [
                [f"{track_root}/{tid}/{name}_keypoints.json" for name in self.img_names]
                for tid in track_ids
            ]
            track_lens = [
                len(list(filter(os.path.isfile, paths))) for paths in track_paths
            ]
            track_ids = [
                track_ids[i]
                for i in np.argsort(track_lens)[::-1]
                if track_lens[i] > MIN_TRACK_LEN
            ]
            print("TRACK LENGTHS", track_ids, track_lens)
            track_ids = track_ids[:n_tracks]
        else:
            track_ids = [f"{int(tid):03d}" for tid in tid_spec.split("-")]

        print("TRACK IDS", track_ids)

        self.track_ids = track_ids
        self.n_tracks = len(track_ids)
        self.track_dirs = [os.path.join(track_root, tid) for tid in track_ids]

        # keep a list of frame index masks of whether a track is available in a frame
        sidx = np.inf
        eidx = -1
        self.track_vis_masks = []
        for pred_dir in self.track_dirs:
            kp_paths = [f"{pred_dir}/{x}_keypoints.json" for x in self.img_names]
            has_kp = [os.path.isfile(x) for x in kp_paths]

            # keep track of which frames this track is visible in
            vis_mask = np.array(has_kp)
            idcs = np.where(vis_mask)[0]
            if len(idcs) > 0:
                si, ei = min(idcs), max(idcs)
                sidx = min(sidx, si)
                eidx = max(eidx, ei)
            self.track_vis_masks.append(vis_mask)

        eidx = max(eidx + 1, 0)
        sidx = min(sidx, eidx)
        print("START", sidx, "END", eidx)
        self.start_idx = sidx
        self.end_idx = eidx
        self.seq_len = eidx - sidx
        self.seq_intervals = [(sidx, eidx) for _ in track_ids]

    def load_data(self):
        self.load_camera_data()
        data_out = {
            "vis_mask": [],
            "joints2d": [],
            "track_interval": []
        }

        T = self.seq_len
        sidx, eidx = self.start_idx, self.end_idx

        for i, tid in enumerate(self.track_ids):
            # load mask of visible frames for this track
            vis_mask = self.track_vis_masks[i][sidx:eidx]  # (T)
            vis_idcs = np.where(vis_mask)[0]
            track_s, track_e = min(vis_idcs), max(vis_idcs) + 1
            data_out["track_interval"].append([track_s, track_e])

            vis_mask = get_ternary_mask(vis_mask)
            data_out["vis_mask"].append(vis_mask)

            # load 2d keypoints for visible frames
            kp_paths = [
                f"{self.track_dirs[i]}/{x}_keypoints.json" for x in self.sel_img_names
            ]
            # (T, J, 3) (x, y, conf)
            joints2d_data = np.stack(
                [read_keypoints(p) for p in kp_paths], axis=0
            ).astype(np.float32)
            # Discard bad ViTPose detections
            joints2d_data[
                np.repeat(joints2d_data[:, :, [2]] < MIN_KEYP_CONF, 3, axis=2)
            ] = 0
            data_out["joints2d"].append(joints2d_data)
        data_out["joints2d"] = np.array(data_out["joints2d"]).transpose(1,0,2,3)

        self.data_dict = data_out

    def get_vis_mask(self):
        return torch.Tensor([item.cpu().numpy() for item in self.data_dict["vis_mask"]])

    def get_track_id(self):
        return torch.Tensor([int(item) for item in self.track_ids]).to(torch.int)

    def load_image(self, path, img_type='rgb'):
        
        if img_type == 'rgb':
            img = imageio.imread(path)
            H, W, c = img.shape
        elif img_type == 'mask':
            img = imageio.imread(path, pilmode='L')
            H, W = img.shape
        if self.resize_hw is not None:
            img = cv2.resize(img, (self.resize_hw[1], self.resize_hw[0]))
        else:
            img = cv2.resize(img, (W, H))
        return img

    def load_camera_data(self, split_cameras=True):
        cam_dir = self.data_sources["cameras"]
        data_interval = 0, -1
        if split_cameras:
            data_interval = self.data_start, self.data_end
        track_interval = self.start_idx, self.end_idx
        self.cam_data = CameraData(
            cam_dir, self.seq_len, self.img_size, data_interval, track_interval
        )

    def get_camera_data(self):
        if self.cam_data is None:
            raise ValueError
        return self.cam_data.as_dict()


    def __len__(self):
        return self.seq_len
    
    def __getitem__(self, idx):
        data = dict()
        joint2d_data = self.data_dict["joints2d"][idx]
        data["joints2d"] = torch.Tensor(joint2d_data)

        data['rgb'] = self.load_image(self.rgb_imgs_path[idx])
        data['mask'] = self.load_image(self.mask_imgs_path[idx], 'mask')

        data["seq_name"] = self.seq_name
        return data
    
class CameraData(object):
    def __init__(
        self, cam_dir, seq_len, img_size, data_interval=[0, -1], track_interval=[0, -1]
    ):
        self.img_size = img_size
        self.cam_dir = cam_dir

        # inclusive exclusive
        data_start, data_end = data_interval
        if data_end < 0:
            data_end += seq_len + 1
        data_len = data_end - data_start

        # start and end indices are with respect to the data interval
        sidx, eidx = track_interval
        if eidx < 0:
            eidx += data_len + 1
        self.sidx, self.eidx = sidx + data_start, eidx + data_start
        self.seq_len = self.eidx - self.sidx

        self.load_data()

    def load_data(self):
        # camera info
        sidx, eidx = self.sidx, self.eidx
        img_w, img_h = self.img_size
        fpath = os.path.join(self.cam_dir, "cameras.npz")
        if os.path.isfile(fpath):
            # Logger.log(f"Loading cameras from {fpath}...")
            cam_R, cam_t, intrins, width, height = load_cameras_npz(fpath)
            scale = img_w / width
            self.intrins = scale * intrins[sidx:eidx]
            t0 = -cam_t[sidx:sidx+1] + torch.randn(3).cpu() * 0.1
            self.cam_R = cam_R[sidx:eidx]
            self.cam_t = cam_t[sidx:eidx] - t0
            self.is_static = False
        else:
            # Logger.log(f"WARNING: {fpath} does not exist, using static cameras...")
            default_focal = 0.5 * (img_h + img_w)
            self.intrins = torch.tensor(
                [default_focal, default_focal, img_w / 2, img_h / 2]
            )[None].repeat(self.seq_len, 1)

            self.cam_R = torch.eye(3)[None].repeat(self.seq_len, 1, 1)
            self.cam_t = torch.zeros(self.seq_len, 3)
            self.is_static = True

        # Logger.log(f"Images have {img_w}x{img_h}, intrins {self.intrins[0]}")
        print("CAMERA DATA", self.cam_R.shape, self.cam_t.shape, self.intrins[0])

    def world2cam(self):
        return self.cam_R, self.cam_t

    def cam2world(self):
        R = self.cam_R.transpose(-1, -2)
        t = -torch.einsum("bij,bj->bi", R, self.cam_t)
        return R, t

    def as_dict(self):
        return {
            "cam_R": self.cam_R,  # (T, 3, 3)
            "cam_t": self.cam_t,  # (T, 3)
            "intrins": self.intrins,  # (T, 4)
            "static": self.is_static,  # bool
        }


def load_cameras_npz(camera_path):
    assert os.path.splitext(camera_path)[-1] == ".npz"

    cam_data = np.load(camera_path)
    height, width, focal = (
        int(cam_data["height"]),
        int(cam_data["width"]),
        float(cam_data["focal"]),
    )
    
    w2c = torch.from_numpy(cam_data["w2c"])  # (N, 4, 4)
    cam_R = w2c[:, :3, :3]  # (N, 3, 3)
    cam_t = w2c[:, :3, 3]  # (N, 3)
    N = len(w2c)

    if "intrins" in cam_data:
        intrins = torch.from_numpy(cam_data["intrins"].astype(np.float32))
    else:
        intrins = torch.tensor([focal, focal, width / 2, height / 2])[None].repeat(N, 1)

    print(f"Loaded {N} cameras")
    return cam_R, cam_t, intrins, width, height


def get_ternary_mask(vis_mask):
    # get the track start and end idcs relative to the filtered interval
    vis_mask = torch.as_tensor(vis_mask)
    vis_idcs = torch.where(vis_mask)[0]
    track_s, track_e = min(vis_idcs), max(vis_idcs) + 1
    # -1 = track out of scene, 0 = occlusion, 1 = visible
    vis_mask = vis_mask.float()
    vis_mask[:track_s] = -1
    vis_mask[track_e:] = -1
    return vis_mask


if __name__ == "__main__":
    data_root = "xxx/data"
    seq_name = 'follow0'
    data_source = {
        "images": os.path.join(data_root, 'images', seq_name),
        "nerf": os.path.join(data_root, 'nerf', seq_name),
        "shots": os.path.join(data_root, 'slahmr', 'shot_idcs', f"{seq_name}.json"),
        "depths": os.path.join(data_root, 'depths', seq_name),
        "tracks": os.path.join(data_root, 'slahmr', "track_preds", seq_name),
        "cameras": os.path.join(data_root, 'slahmr', "cameras", seq_name, 'shot-0')
    }
    start_id = 0
    end_id = 180
    dataset = CameraTrajectoryDataset(data_source, seq_name, start_id, end_id)
    for i in range(2):
        data = dataset.__getitem__(i)
        depth = data['depth']
        track_id = data["track_id"]
        joints2d = data['joints2d']
        vis_mask = data['vis_mask']
        cv2.imshow('1', depth)
        cv2.waitKey(0)