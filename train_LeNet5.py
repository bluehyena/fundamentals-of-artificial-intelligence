import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.LeNet5 import LeNet5

def main():
    lr = 1e-3
    batch_size = 32
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # input size 에 맞게 변형
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    # 데이터셋 정의
    trainset = torchvision.datasets.MNIST(root='./datasets',
                                        transform=transform, train=True, download=True) # 60000장
    testset = torchvision.datasets.MNIST(root='./datasets',
                                        transform=transform, train=False, download=True) # 10000장

    # 데이터 로더 정의
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # 모델
    model = LeNet5().to(device)

    # Loss, Optimizer
    criterion = nn.CrossEntropyLoss() # 논문에서는 MSE
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 학습
    for epoch in range(epochs):    
        for images, labels in train_loader:       
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)              
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad() # gradient 값을 0으로 변경 
            loss.backward() # backpropagation, loss function의 gradient값을 .grad에 저장
            optimizer.step() # 계산된 gradient로 매개변수 업데이트
        
        print(f"Epoch [{epoch}/{epochs}] Loss: {loss.item():.4f}")

    print("\n학습완료\n")

    # 예측결과 확인
    model.to('cpu')

    for i in range(10):
        pred = torch.argmax(model(trainset[i][0])).item()
        label = trainset.targets[i]
        
        print(f"모델 예측값 : {pred}, 실제 값 : {label}")

if __name__ == "__main__":
    main()