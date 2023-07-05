import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data

CONFIGURES = {
    "VGG11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "VGG13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "VGG19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGGNet(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True, vgg_name: str = "VGG19") -> None:
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        self.features = self._make_layers(CONFIGURES[vgg_name], batch_norm=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7,7))
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        if init_weights:
            self._init_weight()

    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu") # fan out: neurons in output layer 이게 뭔뜻일까..
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

    def _make_layers(self, CONFIGURES:list, batch_norm: bool = False) -> nn.Sequential:
        layers: list = []
        in_channels = 3
        for value in CONFIGURES:
            if value == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels=in_channels, out_channels=value, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(value), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

                in_channels = value
        return nn.Sequential(*layers)