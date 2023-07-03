import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.AlexNet import AlexNet




def main():

    # define pytorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define model parameters
    NUM_EPOCHS = 50
    BATCH_SIZE = 64
    MOMENTUM = 0.9
    LR_DECAY = 0.0005
    LR_INIT = 0.01
    IMAGE_DIM = 227  # pixels
    NUM_CLASSES = 1000  # 1000 classes for imagenet 2012 dataset
    DEVICE_IDS = [0]  # GPUs to use

    TRAIN_IMG_DIR = "./datasets"

    seed = torch.initial_seed()
    model = AlexNet(num_classes=NUM_CLASSES).to(device)
    model = torch.nn.parallel.DataParallel(model, divice_ids=DEVICE_IDS)
    print(model)

    # dataset, dataloader
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    ]))

    dataloader = data.DataLoader(
        dataset=dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE
    )

    # optimizer
    optimizer = optim.SGD(
        params=model.parameters(),
        lr = LR_INIT,
        momentum=MOMENTUM,
        weight_decay=LR_DECAY
    )

    # lr scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # model training
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step()

        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            output = model(imgs)
            loss = F.cross_entropy(output, classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_steps += 1

if __name__ == "__main__":
    main()