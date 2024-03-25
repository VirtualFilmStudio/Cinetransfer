# Cinematic Transfer

The repo is the official implemenation of **Cinematic Behavior Transfer via NeRF-based Differentiable Filming**.

[[Project Homepage]( https://virtualfilmstudio.github.io/projects/cinetransfer/)] [[arxiv](https://arxiv.org/pdf/2311.17754.pdf)]

We have **[VirtualFilmStudio](https://virtualfilmstudio.github.io/)** to showcase some of our explorations in filmmaking directions.

## News
- **[2024.02.xx]** We release our code.
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
|-CAMERA_OPTIM
|---camera_optim
|---data
|---torch_ngp
|---slahmr
|-----demo
|-------videos
|---------arc2.mp4 
|...
```
Preprocess data by using SLAHMR.
```shell
cd {YOUR_ROOT}/CAMERA_OPTIM/slahmr/slahmr

# delete slahmr default test dataset
rm ../demo/videos/022691_mpii_test.mp4

# copy test data to slahmr demo
cp ../../data/videos/arc2.mp4 ../demo/videos/

# preprocess data by using SLAHMR
# If the server is not configured with virtual graphics, it is recommended to set 'run_vis=False'
source activate slahmr && python run_opt.py data=video data.seq=arc2 data.root={YOUR_ROOT}/CAMERA_OPTIM/slahmr/demo run_opt=True run_vis=True
```
After execution, an outputs folder will be generated. You can find a folder starting with arc2, which contains optimized smpl.

## 2 Prepare camera optimization data
```shell
mkdir {YOUR_ROOT}/CAMERA_OPTIM/data/arc2

cd {YOUR_ROOT}/CAMERA_OPTIM/camera_optim

# Copy the slahmr preprocessed data to the CAMERA_OPTIM/data folder and adjust the data structure
source activate slahmr && python trans_slahmr_to_nerf.py --src_root {YOUR_ROOT}/CAMERA_OPTIM/slahmr/demo --dst_root {YOUR_ROOT}/CAMERA_OPTIM/data

# Generate depth data
source activate slahmr && python filmingnerf/preproc/extract_depth.py --data data --seq_name arc2 --data_path {YOUR_ROOT}/CAMERA_OPTIM

# Generate mask data
source activate slahmr && python filmingnerf/preproc/gen_mask.py --data data --seq_name arc2 --data_path {YOUR_ROOT}/CAMERA_OPTIM
```
Copy `{YOUR_ROOT}/CAMERA_OPTIM/slahmr/outputs/.../arc2-xxxx/motion_chunks/arc2_{MAX_NUM}_world_results.npz` to the `{YOUR_ROOT}/CAMERA_OPTIM/data/arc2/` folder and rename it to `arc2.npz`.


## 3 Optimize camera
### 3.1 Install torch-ngp and related env

```shell
conda deactivate
cd {YOUR_ROOT}/CAMERA_OPTIM
git clone --recursive https://github.com/ashawkey/torch-ngp.git
mv torch-ngp torch_ngp
cd torch_ngp

conda create -n camera python=3.9 -y
source activate camera && pip install -r requirements.txt

source activate camera && conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# install the tcnn backbone
source activate camera && pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# install all extension modules
bash scripts/install_ext.sh

cd raymarching
source activate camera && python setup.py build_ext --inplace
# install to python path (you still need the raymarching/ folder, since this only install the built extension.)
source activate camera && pip install . 
```
Notes:
- If you have any problems installing ```torch-ngp``` environment .You can refer to [torch-ngp](https://github.com/ashawkey/torch-ngp).
- Rename ```{YOUR_ROOT}/CAMERA_OPTIM/torch-ngp``` to ```{YOUR_ROOT}/CAMERA_OPTIM/torch_ngp```.

### 3.2 Training d-nerf
```shell
cd {YOUR_ROOT}/CAMERA_OPTIM/camera_optim
source activate camera && sudo apt-get update 
source activate camera && sudo apt install xvfb

source activate camera && pip install einops chardet
source activate camera && pip install pyrender
source activate camera && pip install smplx[all]
source activate camera && pip install imageio[ffmpeg] imageio[pyav]
```
Open ```{YOUR_ROOT}/CAMERA_OPTIM/torch_ngp/dnerf/utils.py```, in the file top add this
```python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

```shell
source activate camera && xvfb-run --auto-servernum --server-num=1 python run_train_nerf.py --seq_name arc2
```

### 3.3 Check first frame pose
By using ```camera_ui``` to optim first_frame. 
```shell
cd {YOUR_ROOT}/CAMERA_OPTIM/camera_optim
source activate camera && pip install gradio
source activate camera && pip install plotly

# temp visual result can be found in {out_dir}/{seq_name}/cache_first_pose.png
source activate camera && python run_camera_ui.py  --seq_name arc2 --out_dir ../data/cam_optim_res
```
When open the UI, please follow those steps to check first pose:
- Step1. Adjust the slide rail on the left to control the Camera pose.
- Step2. Click the 'render' button to view the rendering results.
- Step3. Finally click the 'optim' button to optimize a single frame.

Note: You can repeat the above process until you select the most suitable camera pose.

### 3.4 Optim camera trajectory
```shell
cd {YOUR_ROOT}/CAMERA_OPTIM/camera_optim

source activate camera && python run_camera_opt.py --seq_name arc2 --out_dir ../data/cam_optim_res
```

### 3.5 Visualization
```shell
cd {YOUR_ROOT}/CAMERA_OPTIM/camera_optim
# smooth (optional)
source activate camera && xvfb-run --auto-servernum --server-num=1 python run_camera_smooth.py --seq_name arc2 --out_dir ../data/cam_optim_res

source activate camera && xvfb-run --auto-servernum --server-num=1 python run_camera_vis.py --seq_name arc2 --out_dir ../data/cam_optim_res
```

![arc2](D:\my_papers\cinematic_transfer\cinetransfer-code\results\arc2.gif)

## Citation

If you find this paper and repo useful for your research, please consider citing our paper.

```bibtex
@article{jiang2023cinematic,
      title={Cinematic Behavior Transfer via NeRF-based Differentiable Filming},
      author={Jiang, Xuekun and Rao, Anyi and Wang, Jingbo and Lin, Dahua and Dai, Bo},
      journal={arXiv preprint arXiv:},
      year={2023}
  }
```

## Acknowledgements
The code is based on SLAHMR, torch-ngp, thanks to them!
