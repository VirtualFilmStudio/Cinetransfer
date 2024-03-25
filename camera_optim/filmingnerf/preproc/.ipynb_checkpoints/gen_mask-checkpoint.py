import os
import cv2
import numpy as np
import imageio
from tqdm import tqdm
import argparse

def main(params):
    root, seq_name, start_id, end_id, save_root = params
    os.makedirs(save_root, exist_ok=True)
    file_list = os.listdir(root)
    for i in tqdm(range(start_id, end_id+1)):
        file_name = "{}_{:0>6d}".format(seq_name, i)
        imgs = find_files(root, file_name, file_list)
        sum_image = np.zeros_like(imgs[0])
        for img in imgs[:1]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data", type=str)
    parser.add_argument('--seq_name', default="arc2", type=str)
    parser.add_argument('--data_path', default="CAMERA_OPTIM", type=str)

    args = parser.parse_args()
    data = args.data
    seq_name = args.seq_name
    data_path = args.data_path

    images_root = f'{data_path}/{data}/{seq_name}/images'
    img_num = len(os.listdir(images_root))
    start_id = 1
    end_id = img_num
    root = f'{data_path}/{data}/{seq_name}/phalp_out'
    save_path = f"{data_path}/{data}/{seq_name}/masks"
    params = [root, seq_name, start_id, end_id, save_path]
    main(params)

