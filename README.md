# Cinematic Transfer

The repo is the official implemenation of **Cinematic Behavior Transfer via NeRF-based Differentiable Filming**.

[[Project Homepage]( https://virtualfilmstudio.github.io/projects/cinetransfer/)] [[arxiv](https://arxiv.org/pdf/2311.17754.pdf)]

We have **[VirtualFilmStudio](https://virtualfilmstudio.github.io/)** to showcase some of our explorations in filmmaking directions.

## Getting start
```shell
git clone https://github.com/VirtualFilmStudio/Cinetransfer.git
```
## 1 SMPL Visualization
We recommend you use anaconda to run our CinematicTransfer. 
### 1.1 Install SLAHMR
Please refer to [SLAHMR](https://github.com/vye16/slahmr/tree/release) work to install env.
Notes: 

- SLAHMR project have update, please use [release branch](https://github.com/vye16/slahmr/tree/release). 
- Use [SLAHMR - Google Colab](https://colab.research.google.com/drive/1IFvek5DSgKb80vtSvXAXh1xmBFMJuxeL?usp=sharing) version to install  is better. Conda env name will set to slahmr.
### 1.2 Preprocess data
Prepare test data. Please place data according to the following framework. 
```shell
|-Cinetransfer
| |-camera_optim
| |-data
| |-torch_ngp
| |-slahmr
| | |-demo
| | | |-videos
| | | | |-arc2.mp4 
|...
```
Preprocess data by using SLAHMR.
```shell
cd {YOUR_ROOT}/Cinetransfer/slahmr/slahmr

# delete slahmr default test dataset
rm ../demo/videos/022691_mpii_test.mp4

# copy test data to slahmr demo
cp ../../data/videos/arc2.mp4 ../demo/videos/

# preprocess data by using SLAHMR
# If the server is not configured with virtual graphics, it is recommended to set 'run_vis=False'
source activate slahmr && python run_opt.py data=video data.seq=arc2 data.root={YOUR_ROOT}/Cinetransfer/slahmr/demo run_opt=True run_vis=True
```
After execution, an outputs folder will be generated. You can find a folder starting with arc2, which contains optimized smpl.

## 2 Prepare camera optimization data
```shell
mkdir {YOUR_ROOT}/Cinetransfer/data/arc2

cd {YOUR_ROOT}/Cinetransfer/camera_optim

# Copy the slahmr preprocessed data to the Cinetransfer/data folder and adjust the data structure
source activate slahmr && python trans_slahmr_to_nerf.py --src_root {YOUR_ROOT}/Cinetransfer/slahmr/demo --dst_root {YOUR_ROOT}/Cinetransfer/data

# Generate mask data
source activate slahmr && python filmingnerf/preproc/gen_mask.py --data data --seq_name arc2 --data_path {YOUR_ROOT}/Cinetransfer
```

Copy `{YOUR_ROOT}/Cinetransfer/slahmr/outputs/.../arc2-xxxx/motion_chunks/arc2_{MAX_NUM}_world_results.npz` to the `{YOUR_ROOT}/Cinetransfer/data/arc2/` folder and rename it to `arc2.npz`.


## 3 Optimize camera
### 3.1 Install torch-ngp and related env

```shell
conda deactivate
cd {YOUR_ROOT}/Cinetransfer
git clone --recursive https://github.com/ashawkey/torch-ngp.git
mv torch-ngp torch_ngp
cd torch_ngp

conda create -n cinetrans python=3.9 -y
source activate cinetrans && pip install -r requirements.txt

source activate cinetrans && conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# install the tcnn backbone
source activate cinetrans && pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# install all extension modules
bash scripts/install_ext.sh

cd raymarching
source activate cinetrans && python setup.py build_ext --inplace
# install to python path (you still need the raymarching/ folder, since this only install the built extension.)
source activate cinetrans && pip install . 
```
Notes:
- If you have any problems installing ```torch-ngp``` environment .You can refer to [torch-ngp](https://github.com/ashawkey/torch-ngp).
- Rename ```{YOUR_ROOT}/Cinetransfer/torch-ngp``` to ```{YOUR_ROOT}/Cinetransfer/torch_ngp```.

### 3.2 Training d-nerf
```shell
cd {YOUR_ROOT}/Cinetransfer/camera_optim
source activate cinetrans && sudo apt-get update 
source activate cinetrans && sudo apt install xvfb

source activate cinetrans && pip install einops chardet
source activate cinetrans && pip install pyrender
source activate cinetrans && pip install smplx[all]
source activate cinetrans && pip install imageio[ffmpeg] imageio[pyav]
```

Open ```{YOUR_ROOT}/Cinetransfer/torch_ngp/dnerf/utils.py```, in the file top add this

```python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

```shell
source activate cinetrans && xvfb-run --auto-servernum --server-num=1 python run_train_nerf.py --seq_name arc2
```

### 3.3 Check first frame pose
By using ```camera_ui``` to optim first_frame. 
```shell
cd {YOUR_ROOT}/Cinetransfer/camera_optim
source activate cinetrans && pip install gradio
source activate cinetrans && pip install plotly

# temp visual result can be found in {out_dir}/{seq_name}/cache_first_pose.png
source activate cinetrans && python run_camera_ui.py  --seq_name arc2 --out_dir ../data/cam_optim_res
```
When open the UI, please follow those steps to check first pose:
- Step1. Adjust the slide rail on the left to control the Camera pose.
- Step2. Click the 'render' button to view the rendering results.
- Step3. Finally click the 'optim' button to optimize a single frame.

Note: You can repeat the above process until you select the most suitable camera pose.

### 3.4 Optim camera trajectory
```shell
cd {YOUR_ROOT}/Cinetransfer/camera_optim

source activate cinetrans && python run_camera_opt.py --seq_name arc2 --out_dir ../data/cam_optim_res
```

### 3.5 Visualization
```shell
cd {YOUR_ROOT}/Cinetransfer/camera_optim
# smooth (optional)
source activate cinetrans && xvfb-run --auto-servernum --server-num=1 python run_camera_smooth.py --seq_name arc2 --out_dir ../data/cam_optim_res

source activate cinetrans && xvfb-run --auto-servernum --server-num=1 python run_camera_vis.py --seq_name arc2 --out_dir ../data/cam_optim_res
```

![arc2](https://github.com/VirtualFilmStudio/Cinetransfer/blob/main/results/arc2.gif)

## Citation

If you find this paper and repo useful for your research, please consider citing our paper.

```bibtex
@article{jiang2023cinematic,
      title={Cinematic Behavior Transfer via NeRF-based Differentiable Filming},
      author={Jiang, Xuekun and Rao, Anyi and Wang, Jingbo and Lin, Dahua and Dai, Bo},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2024}
  }
```

## Acknowledgements
The code is based on SLAHMR, torch-ngp, thanks to them!
