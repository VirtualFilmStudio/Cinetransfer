import numpy as np
import json
import imageio
from einops import rearrange
import cv2
import os

OP_NUM_JOINTS = 25

def read_keypoints(keypoint_fn):
    """
    Only reads body keypoint data of first person.
    """
    empty_kps = np.zeros((OP_NUM_JOINTS, 3), dtype=np.float)
    if not os.path.isfile(keypoint_fn):
        return empty_kps

    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    if len(data["people"]) == 0:
        print("WARNING: Found no keypoints in %s! Returning zeros!" % (keypoint_fn))
        return empty_kps

    person_data = data["people"][0]
    body_keypoints = np.array(person_data["pose_keypoints_2d"], dtype=np.float)
    body_keypoints = body_keypoints.reshape([-1, 3])
    return body_keypoints


def read_image(img_path, img_wh, blend_a=True):
    img = imageio.imread(img_path).astype(np.float32)/255.0
    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4: # blend A to RGB
        if blend_a:
            img = img[..., :3]*img[..., -1:]+(1-img[..., -1:])
        else:
            img = img[..., :3]*img[..., -1:]

    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> (h w) c')

    return img