import os
import cv2
import numpy as np
import imageio

def main(params):
    mask_root, track_root, seq_name, start_id, end_id, save_root = params
    save_path = os.path.join(save_root, seq_name)
    track_path = os.path.join(track_root, seq_name)
    os.makedirs(save_path, exist_ok=True)
    file_list = os.listdir(mask_root)

    for i in range(start_id, end_id+1):
        file_name = "{}_{:0>6d}".format(seq_name, i)
        imgs = find_files(mask_root, file_name, file_list)
        sum_image = np.zeros_like(imgs[0])
        for img in imgs:
            sum_image = cv2.add(img, sum_image)
        save_img_path = os.path.join(save_path, "{:0>6d}.jpg".format(i))
        cv2.imwrite(save_img_path, sum_image)


def find_files(root, file_name, file_list):
    out = []
    for i in file_list:
        if i.startswith(file_name):
            if 'png' in i:
                img_path = os.path.join(root, i)
                img = imageio.imread(img_path)
                out.append(img)
    return out

if __name__ == '__main__':
    mask_root = 'D:/Python Code/CameraExtractor/data/slahmr/phalp_out/_TMP'
    track_root = 'D:/Python Code/CameraExtractor/data/slahmr/track_preds/'
    seq_name = 'follow0'
    start_id = 1
    end_id = 821
    save_path = "D:/Python Code/CameraExtractor/data/masks_color"
    params = [mask_root, track_root, seq_name, start_id, end_id, save_path]
    main(params)