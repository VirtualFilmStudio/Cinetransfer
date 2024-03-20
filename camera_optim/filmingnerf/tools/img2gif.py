import imageio
import os

img_root = '../output/push0'
end_id = 57
imgs = []
for i in range(0, end_id):
    img_path = os.path.join(img_root, str(i), '190_rgb.png')
    img = imageio.v2.imread(img_path)
    imgs.append(img)


imageio.mimwrite(os.path.join(img_root, 'rgb_vid_src.gif'), imgs, fps=30)