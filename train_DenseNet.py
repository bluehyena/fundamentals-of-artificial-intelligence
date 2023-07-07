# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from models.DenseNet import DenseNet_121
from torch.optim.lr_scheduler import ReduceLROnPlateau

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt


# utils
import numpy as np
import time
import copy

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    random_seed = 123
    initial_lr = 0.01
    num_epoch = 50

    train_data = datasets.STL10('./datasets', split='train', download=True, transform=transforms.ToTensor())
    val_data = datasets.STL10('./datasets', split='test', download=True, transform=transforms.ToTensor())
    
    # define transformation
    transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(64)
    ])

    # apply transformation to dataset
    train_data.transform = transformation
    val_data.transform = transformation

    # make dataloade
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    
    # model
    model = DenseNet_121()
    
    # define loss function, optimizer, lr_scheduler
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=initial_lr)

    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=8)
    
    # get current lr
    def get_lr(opt):
        for param_group in opt.param_groups:
            return param_group['lr']


    # calculate the metric per mini-batch
    def metric_batch(output, target):
        pred = output.argmax(1, keepdim=True)
        corrects = pred.eq(target.view_as(pred)).sum().item()
        return corrects


    # calculate the loss per mini-batch
    def loss_batch(loss_func, output, target, opt=None):
        loss_b = loss_func(output, target)
        metric_b = metric_batch(output, target)

        if opt is not None:
            opt.zero_grad()
            loss_b.backward()
            opt.step()
        
        return loss_b.item(), metric_b

    # calculate the loss per epochs
    def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
        running_loss = 0.0
        running_metric = 0.0
        len_data = len(dataset_dl.dataset)

        for xb, yb in dataset_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)

            loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

            running_loss += loss_b

            if metric_b is not None:
                running_metric += metric_b
            
            if sanity_check is True:
                break

            loss = running_loss / len_data
            metric = running_metric / len_data

            return loss, metric
        
    # function to start training
    def train_val(model, params):
        num_epochs=params['num_epochs']
        loss_func=params['loss_func']
        opt=params['optimizer']
        train_dl=params['train_dl']
        val_dl=params['val_dl']
        sanity_check=params['sanity_check']
        lr_scheduler=params['lr_scheduler']
        path2weights=params['path2weights']

        loss_history = {'train': [], 'val': []}
        metric_history = {'train': [], 'val': []}

        best_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        start_time = time.time()

        for epoch in range(num_epochs):
            current_lr = get_lr(opt)
            print('Epoch {}/{}, current lr= {}'.format(epoch, num_epochs-1, current_lr))

            model.train()
            train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
            loss_history['train'].append(train_loss)
            metric_history['train'].append(train_metric)

            model.eval()
            with torch.no_grad():
                val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
            loss_history['val'].append(val_loss)
            metric_history['val'].append(val_metric)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights)
                print('Copied best model weights!')

            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                print('Loading best model weights!')
                model.load_state_dict(best_model_wts)

            print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
            print('-'*10)

        model.load_state_dict(best_model_wts)
        return model, loss_history, metric_history
    
    # define the training parameters
    params_train = {
        'num_epochs':30,
        'optimizer':opt,
        'loss_func':loss_func,
        'train_dl':train_dl,
        'val_dl':val_dl,
        'sanity_check':False,
        'lr_scheduler':lr_scheduler,
        'path2weights':'./models/weights.pt',
    }

    # check the directory to save weights.pt
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSerror:
            print('Error')
            
    createFolder('./models')

    model, loss_hist, metric_hist = train_val(model, params_train)

if __name__ == "__main__":
    main()