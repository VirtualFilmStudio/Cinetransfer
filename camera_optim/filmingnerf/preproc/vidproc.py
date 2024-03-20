import mmcv
import os
import cv2


def imgs2vid(imgs_path, img_file_tmpl, video_path):
    mmcv.frames2video(imgs_path, video_path, filename_tmpl=img_file_tmpl)

def vid2imgs(video_path, imgs_save_path):
    video = mmcv.VideoReader(video_path)
    video.cvt2frames(imgs_save_path)


if __name__ == '__main__':
    # root = "D:/BaiduNetdiskDownload/posetrack/PoseTrack2018/posetrack18_images.tar/images/val"
    # filename_tmpl='{:06d}.jpg'
    # save_root = "D:/BaiduNetdiskDownload/posetrack/PoseTrack2018/videos"

    # file_list = os.listdir(root)
    # for file_name in file_list:

    #     imgs_path = os.path.join(root, file_name)
    #     video_path = os.path.join(save_root, f'{file_name}.mp4')

    #     mmcv.frames2video(imgs_path, video_path, filename_tmpl=filename_tmpl)

    video_path = 'D:/Python Code/CameraExtractor/blender_out/lalaland-single.mp4'
    save_path = 'D:/Python Code/CameraExtractor/blender_out/lalaland-single'
    os.makedirs(save_path, exist_ok=True)
    vid2imgs(video_path, save_path)