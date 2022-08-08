import torch
from torch import nn
import torch.nn.functional as F

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(1,28,28)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0), #output_shape=(16,24,24)
            nn.ReLU(), # activation
            # Max pool 1
            nn.MaxPool2d(kernel_size=2), #output_shape=(16,12,12)
            # Convolution 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0),#output_shape=(32,8,8)
            nn.ReLU(), # activation
            # Max pool 2
            nn.MaxPool2d(kernel_size=2) #output_shape=(32,4,4)
        )
        self.linear = nn.Sequential(
            # Fully connected 1 ,#input_shape=(32*4*4)
            nn.Linear(32 * 4 * 4, 32), 
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
