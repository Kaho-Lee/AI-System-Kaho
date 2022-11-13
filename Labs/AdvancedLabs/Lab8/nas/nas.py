import argparse
import functools
import logging
import os
import pprint
import random
import sys
import json

import model
import numpy as np
import nni

from nni.retiarii.oneshot.pytorch import DartsTrainer
from nni.retiarii import fixed_arch
from nni.nas.pytorch.fixed import apply_fixed_architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import SubsetRandomSampler

def reset_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except:
        # maybe not available
        pass

def data_preprocess(args):
    def cutout_fn(img, length):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

    augmentation = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    cutout = [functools.partial(cutout_fn, length=args.cutout)] if args.cutout else []
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

    transform_train = transforms.Compose(augmentation + normalize + cutout)
    transform_test = transforms.Compose(normalize)

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset

def accuracy(output, target):
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()

    accuracy = dict()
    accuracy["acc1"] = 100. * correct / target.size(0)

    return accuracy

def main_search(args):
    reset_seed(args.seed)
    gpu_num = 1 if torch.cuda.is_available() else 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_space = model.CNN(32, 3, 8, 10, 8) #From NNI exmaple
    trainset,testset = data_preprocess(args)
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model_space.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

    trainer = DartsTrainer(
        model=model_space,
        loss=criterion,
        metrics=lambda output, target: accuracy(output, target),
        optimizer=optim,
        dataset=trainset,
        batch_size=args.batch_size,
        log_frequency=args.log_frequency,
        num_epochs=args.epochs,
        device=device,
        workers=args.num_workers,
        unrolled=False
    )
    trainer.fit()

    final_architecture = trainer.export()
    print('Final architecture:', final_architecture)
    json.dump(trainer.export(), open('checkpoint_mut.json', 'w'))


def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_frequency == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
      correct, len(test_loader.dataset), accuracy))

    return accuracy

def main_retrain(args):
    print("Model Retrain")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainset,testset = data_preprocess(args)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)
    with fixed_arch("./checkpoint_mut.json"):
      model_darts = model.CNN(32, 3, 16, 10, 20).to(device)
    #apply_fixed_architecture(model_darts, "./checkpoint_mut.json")
    optimizer = torch.optim.SGD(model_darts.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1E-6)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        drop_prob = 0.2 * epoch / args.epochs
        model_darts.drop_path_prob(drop_prob)

        train(args, model_darts, device, train_loader, optimizer, criterion, epoch)
        test(model_darts, device, test_loader, criterion)
        lr_scheduler.step()

    if args.save_model:
        torch.save(model_darts.state_dict(), "darts_cifar10.pt")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 DARTS Training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--cutout', default=0, type=int, help='cutout length in data augmentation')
    parser.add_argument('--epochs', default=8, type=int, help='epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--seed', default=42, type=int, help='global initial seed')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers to preprocess data')
    parser.add_argument('--log_frequency', default=10, type=int, help='number of mini-batches between logging')
    parser.add_argument('--search', default=False, type=bool, help='number of mini-batches between logging')
    parser.add_argument('--save_model', default=False, type=bool, help='number of mini-batches between logging')

    args = parser.parse_args()
    if args.search:
      main_search(args)
    else:
      main_retrain(args)