import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Feature Extractor 부분
        self.net = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inpace=True), # inplace는 input으로 들어온 것 자체를 수정하겠다는 뜻 임
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), 
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )

        # Classifier (FCNN 부분)
        self.classifier = nn.Sequential(
            # FC1
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(inplace=True),
            # FC2
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Output
            nn.Linear(4096, num_classes)
        )
        
        # bias, weight initialization
        def init_bias_weights(self):
            for layer in self.net:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01) # weight 초기화
                    nn.init.constant_(layer.bias, 0) # bias 초기화 0

            # Conv2,4,5 는 bias 1로 초기화
            nn.init.constant_(self.net[4].bias, 1)
            nn.init.constant_(self.net[10].bias, 1)
            nn.init.constant_(self.net[12].bias, 1)

        def forward(self, x):
            x = self.net(x)
            x = x.view(-1, 256*6*6) # 텐서크기 변경 2d
            return self.classifier(x)