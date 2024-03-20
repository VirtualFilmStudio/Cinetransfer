import mmcv
import cv2

video_path = '../data/video/follow0.mp4'
out_dir = '../data/images/follow0'

video = mmcv.VideoReader(video_path)
video.cvt2frames(out_dir)