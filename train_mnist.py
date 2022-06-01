import argparse
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mnist_net import MNISTNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, optimizer, loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')

def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total

if __name__=='__main__':

  parser = argparse.ArgumentParser()
  
  parser.add_argument('--exp_name', type=str, default = 'MNIST', help='experiment name')
  parser.add_argument(...)
  parser.add_argument(...)
  parser.add_argument(...)

  args = parser.parse_args()
  exp_name = args.exp_name
  epochs = ...
  batch_size = ...
  lr = ...

  # transforms
  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))])

  # datasets
  trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
  testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)

  # dataloaders
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
  
  net = ...
  # setting net on device(GPU if available, else CPU)
  net = net.to(device)
  optimizer = optim.SGD(...)

  train(...)
  test_acc = test(...)
  print(f'Test accuracy:{test_acc}')
  torch.save(net.state_dict(), "mnist_net.pth")
