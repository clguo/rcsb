import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .base import ClassifierBase

class LeNet(ClassifierBase):
    def __init__(self, path_ckpt: str = None):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

        self.transforms = transforms.Compose([
            transforms.CenterCrop((28, 28)),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),  # MNIST mean and std
        ])

        self.load_state_dict(torch.load(path_ckpt))

    def forward(self, x):
        x = self.transforms(x)

        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)

        y = y.view(y.shape[0], -1)

        y = self.fc1(y)
        y = self.relu3(y)

        y = self.fc2(y)
        y = self.relu4(y)

        y = self.fc3(y)
        return y

    def pred_prob(self, x):
        x = self(x)
        return F.softmax(x, dim = 1)

    def pred_label(self, x):
        x = self.pred_prob(x)
        return x.argmax(dim = 1).long()

    def load_ckpt(self, *args, **kwargs):
        pass

    def use_functional_relu_only(self):
        pass