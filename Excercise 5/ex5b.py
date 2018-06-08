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


def train(loader, phase):
    NET.train() if phase == "train" else NET.eval()
    correct = loss = avg_loss = 0
    for data, labels in loader:
        if phase == "train":
            OPTIMIZER.zero_grad()

        output = NET(data)
        loss = CRITERION(output, labels)
        avg_loss += loss.item()
        if phase == "train":
            loss.backward()
            OPTIMIZER.step()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()

    # print statistics
    batch_size = BATCH_SIZE if phase == "train" else 1
    avg_loss /= (len(loader.sampler))
    print "%s loss: %-12.9f %s accuracy: %-8s" % (
        phase, avg_loss, phase, str((100. * correct / (len(loader) * batch_size))) + "%")
    return avg_loss


def start_learning(train_loader, validation_loader):
    loss = {}
    phase_loss = {}
    tes = {}
    for epoch in range(EPOCHS):
        print "Starting epoch: %d" % epoch
        for phase in ["train", "eval"]:
            loader = train_loader if phase == "train" else validation_loader
            phase_loss[epoch] = train(loader, phase)
            loss[phase] = phase_loss
    print "Finished training"
    return loss


def pred_and_write(validation_loader):
    loss = 0.0
    with open("test.pred", "w") as t, open("real.pred", "w") as r:
        for data, target in validation_loader:
            output = NET(data)
            loss += F.nll_loss(output, target, size_average=False)
            pred = output.data.max(1, keepdim=True)[1]
            t.write(str(pred.item()) + "\n")
            r.write(str(target.item()) + "\n")

    loss /= len(validation_loader)
    print "Validation loss: " + str(loss.item())


def compare(a="test.pred", b="real.pred"):
    correct = total = 0.0
    with open(a, "r") as t, open(b, "r") as r:
        for x, y in zip(t, r):
            total += 1
            if x == y: correct += 1
    print "Validation accuracy: " + str(100. * correct / total) + "%"


def plot_loss(losses, title):
    plt.rcParams.update({'font.size': 12})
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.box(False)
    plt.minorticks_on()
    plt.tick_params(direction='out', color='black')
    plt.grid(color='black', alpha=0.01, linewidth=0.3, which='both')
    plt.plot(losses["train"].keys(), losses["train"].values(), color='red', linewidth=5, linestyle='dotted')
    plt.plot(losses["eval"].keys(), losses["eval"].values(), color='purple', linewidth=2)
    plt.legend(('Train loss', 'Test loss'), fancybox=False, edgecolor='white', fontsize='small')
    plt.show()


def load():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(num_train * 0.2)

    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader, validation_loader


BATCH_SIZE = 64
EPOCHS = 10
ETA = 0.001
NET = Net()
OPTIMIZER = optim.SGD(NET.parameters(), lr=ETA, momentum=0.9)
CRITERION = nn.CrossEntropyLoss(size_average=False)


def main():
    train_loader, test_loader, validation_loader = load()
    losses = start_learning(train_loader=train_loader, validation_loader=validation_loader, )
    pred_and_write(test_loader)
    compare()
    plot_loss(losses, "Model C")


if __name__ == '__main__':
    main()
