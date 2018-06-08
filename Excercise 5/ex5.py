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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(net, train_loader, optimizer, batch_size):
    net.train()
    correct = 0
    loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
    # print statistics
    print "Average train loss: %.9f, Train accuracy: %s" % (
        loss, str((100. * correct.item() / (len(train_loader) * batch_size))) + "%")

    return loss.item()


def validate(net, test_loader):
    net.eval()
    loss = correct = 0
    for data, target in test_loader:
        output = net(data)
        loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    loss /= len(test_loader)

    # Print accuracy #
    print("Average validation loss: %.9f, Validation Accuracy: %s" % (
        loss, str((100. * correct.item() / len(test_loader))) + "%"))

    return loss


def start_learning(model, train_loader, optimizer, validation_loader, batch_size, epochs=10):
    test_loss = train_loss = {}

    for epoch in range(epochs):
        print "Starting epoch: %d" % epoch
        train_loss[epoch] = train(model, train_loader, optimizer, batch_size)
        test_loss[epoch] = validate(model, validation_loader)

    print "Finished training"

    return train_loss, test_loss


def pred_and_write(model, validation_loader):
    f = open("test.pred", "w")
    r = open("real.pred", "w")

    loss = 0.0
    for data, target in validation_loader:
        output = model(data)
        loss += F.nll_loss(output, target, size_average=False)
        pred = output.data.max(1, keepdim=True)[1]
        f.write(str(pred.item()) + "\n")
        r.write(str(target.item()) + "\n")

    loss /= len(validation_loader)
    print "Validation loss: " + str(loss.item())

    r.close()
    f.close()


def compare(a="test.pred", b="real.pred"):
    correct = total = 0.0

    f = open(a, "r")
    r = open(b, "r")
    for x, y in zip(f, r):
        total += 1
        if x == y:
            correct += 1
    print "Validation accuracy: " + str(100. * correct / total) + "%"

    f.close()
    r.close()


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


def load(batch_size):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(num_train * 0.2)

    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader, validation_loader


def main():
    BATCH_SIZE = 64
    EPOCHS = 10
    ETA = 0.001
    train_loader, test_loader, validation_loader = load(BATCH_SIZE)
    net = Net()

    optimizer = optim.SGD(net.parameters(), lr=ETA, momentum=0.9)
    train_loss, test_loss = start_learning(model=net, epochs=EPOCHS, train_loader=train_loader,
                                           optimizer=optimizer, validation_loader=validation_loader,
                                           batch_size=BATCH_SIZE)
    pred_and_write(net, test_loader)
    compare()
    plot_loss(train_loss, test_loss, "Model C")


if __name__ == '__main__':
    main()
