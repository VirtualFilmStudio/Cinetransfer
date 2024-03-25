import os
import shutil
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_root', default="../demo10", type=str)
parser.add_argument('--dst_root', default="../data10", type=str)
args = parser.parse_args()

src_root = args.src_root    # '../demo10'
dst_root = args.dst_root    # '../data10'

video_list = os.listdir(os.path.join(src_root, "images"))

for video in tqdm(video_list[:]):
    if video not in ['demo1', 'demo2', 'demo10']:
        camera_root = os.path.join(dst_root, video, 'cameras')
        depth_root = os.makedirs(os.path.join(dst_root, video, 'depths'), exist_ok=True)
        dnerf_root = os.makedirs(os.path.join(dst_root, video, 'dnerf'), exist_ok=True)
        image_root = os.path.join(dst_root, video, 'images')
        mask_root = os.makedirs(os.path.join(dst_root, video, 'masks'), exist_ok=True)
        phalp_root = os.path.join(dst_root, video, 'phalp_out')
        shot_root = os.path.join(dst_root, video, 'shot_dics')
        track_root = os.path.join(dst_root, video, 'track_preds')

        src_path = os.path.join(src_root, 'slahmr', 'cameras', video)
        shutil.copytree(src_path, camera_root)

        src_path = os.path.join(src_root, 'images', video)
        shutil.copytree(src_path, image_root)

        os.makedirs(shot_root, exist_ok=True)
        src_path = os.path.join(src_root, 'slahmr', 'shot_idcs', f'{video}.json')
        dst_path = os.path.join(shot_root, f'{video}.json')
        shutil.copy(src_path, dst_path)

        os.makedirs(shot_root, exist_ok=True)
        src_path = os.path.join(src_root, 'slahmr', 'track_preds', video)
        shutil.copytree(src_path, track_root)

        os.makedirs(phalp_root, exist_ok=True)
        src_path = os.path.join(src_root, 'slahmr', 'phalp_out', '_TMP')
        file_list = os.listdir(src_path)
        for file_name in file_list:
            if video in file_name:
                src_image_path = os.path.join(src_path, file_name)
                dst_path = os.path.join(phalp_root, file_name)
                shutil.copy(src_image_path, dst_path)







