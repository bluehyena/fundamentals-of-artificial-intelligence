from models.ResNet import ResNet
import torch

model = ResNet().ResNet152()
y = model(torch.randn(1, 3, 224, 224))
print (y.size())