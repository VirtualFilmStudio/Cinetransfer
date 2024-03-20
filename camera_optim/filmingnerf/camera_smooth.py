import os
import json
import numpy as np
import pyrender

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from renderer.geometry.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix
from renderer.renderer import Renderer, Transform
from tools.tensor import get_device
from opt import get_opts

def openJson(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def prepare_data(cfg, device):
    data = openJson(f"{cfg.out_dir}/{cfg.seq_name}/optim_cams.json")
    data = torch.Tensor(data).to(device)
    trans = data[:, :3, 3]
    rot_m = data[:, :3, :3]
    return data, trans, rot_m

class CameraTrjactory(nn.Module):
    def __init__(self, b0, b3):
        super(CameraTrjactory, self).__init__()
        self.b0 = b0
        self.b1 = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.b2 = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.b3 = b3

    def forward(self,t):
        t = torch.tensor(t)
        bt = self.b0 * torch.pow(1-t,3) + self.b1 * 3 * t * torch.pow(1-t, 2) + \
                    self.b2 * 3 * torch.pow(t, 2) * (1-t) + self.b3 * torch.pow(t, 3)
        return bt


def run_opt_position(cfg, device):
    camera_poses, camera_trans, _ = prepare_data(cfg, device)
    camTrj = CameraTrjactory(camera_trans[0],camera_trans[-1])
    optimizer = torch.optim.Adam(camTrj.parameters(), weight_decay=0.01)
    step = 1 / (len(camera_trans)-1)
    epochs = 300
    for epoch in range(epochs):
        running_loss = 0.0
        for t in range(len(camera_trans)-2):
            x_ = step * (t+1)
            y_ = camera_trans[t+1]
            
            y_pred = camTrj(x_)
            loss = torch.abs(y_pred - y_).mean()

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch: {epoch + 1:02}/{epochs} Loss: {running_loss:.5e}")

    out_poses = [] 
    renderer = Renderer(150,64,alight=[0.3, 0.3, 0.3],bg_color=[1.0, 1.0, 1.0, 0.0])
    light_trans = Transform((10, 10, 10))
    for t in range(len(camera_poses))[:]:
        renderer.add_light('directlight', light_trans, np.ones(3), 0.9)
        c2w = camera_poses[t].cpu().detach()
        new_trans = camTrj(t*step).cpu().detach()
        c2w[:3, 3] = new_trans
        out_poses.append(c2w.tolist())
        if t%3==0:
            renderer.add_camera(c2w.numpy(), visiable=True)
    # pyrender.Viewer(renderer.scene)
    # import pdb;pdb.set_trace()

    save_root = os.path.join(cfg.out_dir, cfg.seq_name)
    with open(os.path.join(save_root, 'optim_cams_smooth.json'), "w") as f:
        json.dump(out_poses, f, indent=4)


def run_opt_rotation(cfg, device):
    camera_poses, _, rot_matrixes = prepare_data(cfg, device)
    rot_6ds = matrix_to_rotation_6d(rot_matrixes)

    PCA_visualise(rot_6ds.cpu().numpy())
    import pdb;pdb.set_trace()

def PCA_visualise(data):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    projected = pca.fit_transform(data)

    x = projected[:,0]
    y = projected[:,1]
    colors = np.random.rand(len(x))
    plt.scatter(x, y, c=colors, alpha=0.5)
    for i, label in enumerate(range(len(x))):
        plt.annotate(label, (x[i], y[i]))
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.show()
    import pdb;pdb.set_trace()

def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cfg = get_opts()
    device = get_device(0)
    run_opt_position(cfg, device)
    # run_opt_rotation(cfg, device)


if __name__ == "__main__":
    main()
