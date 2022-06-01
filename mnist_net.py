import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(...)
        self.conv2 = nn.Conv2d(...)
        self.pool = nn.MaxPool2d(...)
        self.fc1 = nn.Linear(...)
        self.fc2 = nn.Linear(...)
        self.fc3 = nn.Linear(...)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # First convolution followed by
        x = self.pool(x)                # a relu activation and a max pooling#
        x = ...
        ...
        x = self.fc3(x)
        return x

    def get_features(self, x):
        pass


if __name__=='__main__':
    x = torch.rand(16,1,28,28)
    net = MNISTNet()
    y = net(x)
    assert y.shape == (16,10)