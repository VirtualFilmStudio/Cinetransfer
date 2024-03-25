# coding=gbk
import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

def load_data(filepath):
    data = np.loadtxt(filepath)
    T = len(data)
    x = []
    time_step = 1 / (T-1)
    for i in range(T):
        x.append(i*time_step)
    
    cam_points = np.array(data)[:,3:]
    y = []
    for cam in cam_points:
        a = cam[0] - cam_points[0][0]
        b = cam[1] - cam_points[0][1]
        c = cam[2] - cam_points[0][2]
        y.append([a,b,c])

    return Variable(torch.Tensor(x)), Variable(torch.Tensor(y))


def train():
    file_path = "D:/Python Code/CameraExtractor/camera_data.txt"
    x, y = load_data(file_path)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    xs = y[:, 0]
    ys = y[:, 1]
    zs = y[:, 2]
    ax.scatter(xs.numpy(), ys.numpy(), zs.numpy(), marker="^")
        
    # plt.show()
    # import pdb;pdb.set_trace()
    print('------      build net      ------')
    class Net(torch.nn.Module):
        def __init__(self, max_len, hidden_dim=512):
            super(Net,self).__init__()
            self.encoder = torch.nn.Linear(1, hidden_dim)
            self.predict = torch.nn.Linear(hidden_dim, 3)
            self.pe = PositionalEncoder(hidden_dim, max_len)

        def forward(self, x):
            x = x.float().unsqueeze(1)
            # import pdb;pdb.set_trace()
            x = self.encoder(x)
            # x = self.pe(x).squeeze(1)
            x = self.predict(F.relu(x))
            return x
    net=Net(max_len=len(x))
 
    print('------      start train      ------')
    loss_func=F.mse_loss
    optimizer=torch.optim.SGD(net.parameters(),lr=0.001)
 

    for t in range(200000):
        prediction=net(x)
        # import pdb;pdb.set_trace()
        loss=loss_func(prediction,y)
        optimizer.zero_grad()  
        loss.backward() 
        optimizer.step()  
 
        if t%1==0:
            print(f"loss: {loss}")
            
    xs = prediction[:, 0].detach()
    ys = prediction[:, 1].detach()
    zs = prediction[:, 2].detach()
    ax.scatter(xs.numpy(), ys.numpy(), zs.numpy(), marker="^")
    plt.show()
    print('------      visulise      ------')
 
if __name__=='__main__':
    train()