import mmcv
import os
import cv2

imgs_path = '../data/images/abandonedfactory'
output_path = '../data/video'
video_name = 'abandonedfactory.mp4'

video_path = os.path.join(output_path, video_name)

mmcv.frames2video(imgs_path, video_path, filename_tmpl='{:06d}_left.png')
