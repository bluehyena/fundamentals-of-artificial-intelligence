# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate) -> None:
        # growth rate: k
        super(BottleNeck, self).__init__()

        inner_channels = 4 * growth_rate

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=inner_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=inner_channels, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        return torch.cat([self.shortcut(x), self.residual(x)], 1)
    
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Transition, self).__init__()

        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)
    
class DenseNet(nn.Module):
    def __init__(self, nblocks, growth_rate=12, reduction=0.5, num_classes=10, init_weights=True) -> None:
        super(DenseNet, self).__init__()

        self.growth_rate = growth_rate
        self.inner_channels = 2 * growth_rate
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=self.inner_channels, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2, padding=1)
        )

        self.features = nn.Sequential()

        for i in range(len(nblocks)- 1):
            self.features.add_module('dense_block_{}'.format(i), self._make_dense_block(nblocks[i], self.inner_channels))
            self.inner_channels += growth_rate * nblocks[i]
            out_channels = int(reduction * self.inner_channels)
            self.features.add_module('transition_layer_{}'.format(i), Transition(in_channels=self.inner_channels, out_channels=out_channels))
            self.inner_channels = out_channels
        
        self.features.add_module('dense_block_{}'.format(len(nblocks) - 1),
                                 self._make_dense_block(nblocks[len(nblocks-1)], self.inner_channels))
        self.inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(self.inner_channels))
        self.features.add_module('relu', nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(in_features=self.inner_channels, out_features=num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x
    
    def _make_dense_block(self, nblock, inner_channels):
        dense_block = nn.Sequential()
        for i in range(nblock):
            dense_block.add_module('bottle_neck_layer_{}'.format(i), BottleNeck(in_channels=inner_channels, growth_rate=self.growth_rate))
            inner_channels += self.growth_rate
        return dense_block
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

def DenseNet_121():
    return DenseNet([6,12,24,6])
