# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

# Global definitions used in multiple classes
FIRST_LAYER = 100
SECOND_LAYER = 50
OUTPUT = 10


class ModelA(nn.Module):
    def __init__(self, image_size=784):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRST_LAYER)
        self.fc1 = nn.Linear(FIRST_LAYER, SECOND_LAYER)
        self.fc2 = nn.Linear(SECOND_LAYER, OUTPUT)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelB(nn.Module):
    def __init__(self, image_size=784):
        super(ModelB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRST_LAYER)
        self.fc1 = nn.Linear(FIRST_LAYER, SECOND_LAYER)
        self.fc2 = nn.Linear(SECOND_LAYER, OUTPUT)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelC(nn.Module):
    def __init__(self, image_size=784):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRST_LAYER)
        self.fc1 = nn.Linear(FIRST_LAYER, SECOND_LAYER)
        self.fc2 = nn.Linear(SECOND_LAYER, OUTPUT)
        self.batch1 = nn.BatchNorm1d(FIRST_LAYER)
        self.batch2 = nn.BatchNorm1d(SECOND_LAYER)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.batch1(self.fc0(x)))
        x = F.relu(self.batch2(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(epoch, model, train_loader, optimizer, batch_size):
    model.train()
    correct = 0
    loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
    print "│%6d │ %-15.9f │ %-16s│" % (
        epoch, loss, str((100. * correct.item() / (len(train_loader) * batch_size))) + "%")
    return loss.item()


def validate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader)
    print("│%6s │ %-15.9f │ %-16s│" % ("", test_loss, str((100. * correct.item() / len(test_loader))) + "%"))
    return test_loss


def start_learning(model, train_loader, optimizer, validation_loader, batch_size, epochs=10):
    print("┌───────┬─────────────────┬─────────────────┐")
    print("│ Epoch │    Avg. loss    │       Acc       │")
    test_loss = {}
    train_loss = {}
    for epoch in range(epochs):
        print("├───────┼─────────────────┼─────────────────┤")
        train_loss[epoch] = train(epoch, model, train_loader, optimizer, batch_size)
        test_loss[epoch] = validate(model, validation_loader)
    print("└───────┴─────────────────┴─────────────────┘")

    return train_loss, test_loss


def pred_and_write(model, validation_loader):
    f = open("test.pred", "w")
    r = open("real.pred", "w")
    loss = 0.0
    for data, target in validation_loader:
        output = model(data)
        loss += F.nll_loss(output, target, size_average=False)  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        f.write(str(pred.item()) + "\n")
        r.write(str(target.item()) + "\n")
    loss /= len(validation_loader)
    print "Validation loss: " + str(loss.item())
    r.close()
    f.close()


def compare(a="test.pred", b="real.pred"):
    correct = 0.0
    total = 0.0
    f = open(a, "r")
    r = open(b, "r")
    for x, y in zip(f, r):
        total += 1
        if x == y:
            correct += 1
    print "Validation accuracy: " + str(100. * correct / total) + "%"
    f.close()
    r.close()


def load(batch_size):
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms)

    # define our indices -- our dataset has 9 elements and we want a 8:4 split
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(num_train * 0.2)

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # define our samplers -- we use a SubsetRandomSampler because it will return
    # a random subset of the split defined by the given indices without replaf
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader, validation_loader


def plot_loss(train_loss, test_loss, title):
    plt.rcParams.update({'font.size': 12})
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.box(False)
    plt.minorticks_on()
    plt.tick_params(direction='out', color='black')
    plt.grid(color='black', alpha=0.01, linewidth=0.3, which='both')
    plt.plot(list(train_loss.keys()), list(train_loss.values()), color='red', linewidth=2, linestyle='dotted')
    plt.plot(list(test_loss.keys()), list(test_loss.values()), color='purple', linewidth=2)
    plt.legend(('Train loss', 'Test loss'), fancybox=False, edgecolor='white', fontsize='small')
    plt.show()
    pass


def main():
    BATCH_SIZE = 64
    EPOCHS = 10
    IMAGE_SIZE = 784
    ETA = 0.001
    train_loader, test_loader, validation_loader = load(BATCH_SIZE)
    model = ModelB(image_size=IMAGE_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=ETA)
    train_loss, test_loss = start_learning(model=model, epochs=EPOCHS, train_loader=train_loader,
                                           optimizer=optimizer,
                                           validation_loader=validation_loader, batch_size=BATCH_SIZE)
    pred_and_write(model, test_loader)
    compare()
    plot_loss(train_loss, test_loss, "Model C")


if __name__ == '__main__':
    main()
