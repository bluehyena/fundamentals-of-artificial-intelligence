# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim

# Swish Activation Function
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)
    
# SE Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4) -> None:
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1,1))