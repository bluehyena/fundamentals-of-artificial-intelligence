import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
from models.VGGNet import VGGNet

if __name__ == "__main__":
    seed = torch.initial_seed()
    BATCH_SIZE = 256
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    CHECKPOINT_PATH = "./checkpoint/"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model 선언
    model = VGGNet(num_classes=10, init_weights=True, vgg_name="VGG19")

    # preprocess 정의
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    # dataset 정의
    train_dataset = datasets.STL10(root='./datasets', download=True, split='train', transform=preprocess)
    test_dataset = datasets.STL10(root='./datasets', download=True, split='test', transform=preprocess)

    # dataloader 정의
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # optimizer, scheduler 정의
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(lr=LEARNING_RATE, weight_decay=5e-3, params=model.parameters(), momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
    model = torch.nn.parallel.DataParallel(model, device_ids=[0,])

    # 학습
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        for idx, _data in enumerate(train_dataloader, start=0):
            images, labels = _data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == labels)
                    print ('Epoch: {} \tStep: {}\tLoss: {:.4f} \tAccuracy: {}'.format(epoch+1, idx, loss.item(), accuracy.item() / BATCH_SIZE))
                    scheduler.step(loss)

    state = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'seed': seed
    }

    if epoch % 50 == 0:
        torch.save(state, CHECKPOINT_PATH+'model_{}.pth'.format(epoch))