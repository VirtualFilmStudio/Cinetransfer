import mmcv
import os
import cv2


def imgs2vid(imgs_path, img_file_tmpl, video_path):
    mmcv.frames2video(imgs_path, video_path, filename_tmpl=img_file_tmpl)

def vid2imgs(video_path, imgs_save_path):
    video = mmcv.VideoReader(video_path)
    video.cvt2frames(imgs_save_path)


if __name__ == '__main__':
    video_path = 'xxx/lalaland-single.mp4'
    save_path = 'xxx/lalaland-single'
    os.makedirs(save_path, exist_ok=True)
    vid2imgs(video_path, save_path)