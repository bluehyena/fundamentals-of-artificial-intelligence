import torch
from torch import nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.S2 = nn.AvgPool2d(kernel_size=2, stride=2) # stride Default = kernel_size
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.S4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.F6 = nn.Linear(120, 84)
        self.F7 = nn.Linear(84,10)

    def forward(self, x):
        x = self.C1(x)
        x = F.sigmoid(self.S2(x))
        x = self.C3(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.F6(torch.squeeze(x))
        x = self.F7(x)

        return x