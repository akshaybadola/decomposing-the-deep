import json
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np

from util import load_fixing_names


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockA(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlockA, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNetA paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2],
                                                  (0, 0, 0, 0, planes//4, planes//4),
                                                  "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetA(nn.Module):
    def __init__(self, block, num_blocks, num_classes, in_planes=16,
                 pen_filters=64, indices=[], pretrained=False):
        super(ResNetA, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(pen_filters, num_classes)
        self.indices = torch.tensor(indices)
        self.indices.requires_grad = False
        self.num_classes = num_classes
        self.final_weights = nn.Parameter(torch.stack([self.linear.weight.T[indices[i], i]
                                                       for i in range(len(indices))]))
        self.final_bias = nn.Parameter(self.linear.bias)
        self.apply(_weights_init)

    def _init_final_weights(self):
        self.final_weights.data.copy_(torch.stack(
            [self.linear.weight.data.T[self.indices[i], i]
             for i in range(len(self.indices))]))
        self.final_bias.data = self.linear.bias.data.clone()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def head(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def tail(self, x):
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        return self.linear(x)

    def forward_regular(self, x):
        return self.tail(self.head(x))

    def forward_decomposed(self, x):
        x = self.head(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = x[:, self.indices]
        return ((x * self.final_weights).sum(-1) + self.final_bias)


def resnet20(num_classes, pretrained=None, **kwargs):
    if pretrained:
        print("Warning! This model does not support \"pretrained\".")
    return ResNetA(BasicBlockA, [3, 3, 3], num_classes=num_classes, **kwargs)


def get_model(weights_file):
    weights = torch.load(weights_file, map_location="cpu")
    with open("resnet20_cifar-10_indices.json") as f:
        indices = json.load(f)
    keys = [*indices.keys()]
    if isinstance(keys[0], str):
        func = str
    else:
        func = int
    model = resnet20(10, indices=[indices[func(i)] for i in range(10)])
    load_fixing_names(model, weights["state_dict"])
    model._init_final_weights()
    return model
