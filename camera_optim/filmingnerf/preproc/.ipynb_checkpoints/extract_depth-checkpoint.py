from midas import MidasDetector
import imageio
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def resize_image(input_image, resolution=1024):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def extract_img_depth(image_path):
    image = imageio.imread(image_path)
    H, W, C = image.shape
    image = resize_image(image, resolution=H//2)
    input_image = HWC3(image)
    apply_midas = MidasDetector()
    detected_map, _ = apply_midas(input_image)
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    return detected_map

def launch_extract_imgs_depth(img_root, save_root):
    files_list = os.listdir(img_root)
    for file in tqdm(files_list[:180]):
        img_id = file[:-4]
        img_path = os.path.join(img_root, file)
        depth = extract_img_depth(img_path)
        depth_save_path = os.path.join(save_root, '{}.jpg'.format(img_id))
        cv2.imwrite(depth_save_path, depth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data", type=str)
    parser.add_argument('--seq_name', default="arc2", type=str)
    parser.add_argument('--data_path', default="CAMERA_OPTIM", type=str)
    args = parser.parse_args()

    data = args.data
    seq_name = args.seq_name
    data_path = args.data_path

    img_root = f'{data_path}/{data}/{seq_name}/images/'
    depth_save_path = f'{data_path}/{data}/{seq_name}/depths/'
    os.makedirs(depth_save_path, exist_ok=True)
    launch_extract_imgs_depth(img_root, depth_save_path)




