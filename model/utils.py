import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectHead(nn.Module):
    def __init__(self,input_channels,output_channels = None):
        super().__init__()
        self.input_channels = input_channels
        if output_channels is None:
            output_channels = input_channels // 4
        self.output_channels = output_channels
        self.head = nn.Sequential(
            nn.Conv2d(self.input_channels,self.output_channels,1,1,0),
            nn.ReLU(),
            nn.Conv2d(self.output_channels,self.output_channels,1,1,0),
        )
    def forward(self,x):
        return F.normalize(self.head(x),p=2,dim=1)