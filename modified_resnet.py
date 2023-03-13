import json
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import torchvision

import numpy as np

from util import load_fixing_names
from indices import Indices


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class SplitAvgPool(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def forward(self, x):
        x = F.avg_pool2d(x, x.shape[-1]).view(x.shape[0], -1)
        return x


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
        self.fc = self.linear
        self.indices = indices
        self.num_classes = num_classes
        if self.indices:
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

    def forward(self, x):
        return self.tail(self.head(x))

    def forward_decomposed(self, x):
        x = self.head(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = x[:, self.indices]
        return ((x * self.final_weights).sum(-1) + self.final_bias)

    def forward_noise_at_inds_for_label(self, x):
        val_label = getattr(self, "_val_label", None)
        inds = self.indices[-1][val_label]
        invert_inds = set(range(64)) - set(inds)
        invert_inds = np.array([*invert_inds])
        if val_label is None:
            raise AttributeError("Model has no attribute _val_label")
        x = self.head(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x_1 = x.clone()
        x_2 = x.clone()
        x_1[:, inds] = torch.randn(x_1.shape[0], len(inds)).to(x_1.device)
        x_2[:, invert_inds] = torch.randn(x_2.shape[0], len(invert_inds)).to(x_2.device)
        return self.fc(x_1), self.fc(x_2)


def resnet20(num_classes, pretrained=None, **kwargs):
    if pretrained:
        print("Warning! This model does not support \"pretrained\".")
    return ResNetA(BasicBlockA, [3, 3, 3], num_classes=num_classes, **kwargs)


class BasicBlockB(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockB, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlockB only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlockB")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # import pdb; pdb.set_trace()
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetB(torch.nn.Module):
    def __init__(self, block, layers, num_classes=1000, indices=[],
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.inplanes = 64
        self.dilation = 1
        self.indices = indices
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_layer=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       num_layer=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       num_layer=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       num_layer=4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.indices:
            self.split_pool = SplitAvgPool(self.indices)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockB):
                    nn.init.constant_(m.bn2.weight, 0)
        if self.indices:
            self.final_weights = nn.Parameter(torch.stack([self.fc.weight.T[self.indices[-1][i], i]
                                                           for i in range(len(self.indices[-1]))]))
            self.final_bias = nn.Parameter(self.fc.bias)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, num_layer=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            num_layer=num_layer, num_block=0))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                num_layer=num_layer, num_block=i))
        return nn.Sequential(*layers)

    def _init_final_weights(self):
        if self.indices:
            self.final_weights.data.copy_(torch.stack(
                [self.fc.weight.data.T[self.indices[i], i]
                 for i in range(len(self.indices))]))
            self.final_bias.data = self.fc.bias.data.clone()

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def head(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_noise_at_inds_for_label(self, x):
        val_label = getattr(self, "_val_label", None)
        inds = self.indices[-1][val_label]
        invert_inds = set(range(2048)) - set(inds)
        invert_inds = np.array([*invert_inds])
        if val_label is None:
            raise AttributeError("Model has no attribute _val_label")
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x_1 = x.clone()
        x_2 = x.clone()
        x_1[:, inds] = torch.randn(x_1.shape[0], len(inds)).to(x_1.device)
        x_2[:, invert_inds] = torch.randn(x_2.shape[0], len(invert_inds)).to(x_2.device)
        return self.fc(x_1), self.fc(x_2)

    def forward_decomposed(self, x):
        x = self.head(x)
        x = self.split_pool(x)
        x = x.view(x.size(0), -1)
        x = x[:, self.indices[-1]]
        return ((x * self.final_weights).sum(-1) + self.final_bias)

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetB(block, layers, **kwargs)
    if pretrained:
        if getattr(torchvision.models, "util", None):
            func = torchvision.models.util.load_state_dict_from_url
        elif hasattr(torchvision, "_internally_replaced_utils"):
            func = torchvision._internally_replaced_utils.load_state_dict_from_url
        else:
            try:
                from torch.hub import load_state_dict_from_url
            except ImportError:
                from torch.utils.model_zoo import load_url as load_state_dict_from_url
            func = load_state_dict_from_url
        print(f"Loading pretrained model {arch}")
        state_dict = func(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def get_model(model_name, weights_file, inds_file, **kwargs):
    indices = inds_file and Indices(inds_file)
    if model_name == "resnet20":
        weights = torch.load(weights_file, map_location="cpu")
        model = resnet20(10, indices=indices, **kwargs)
        if "state_dict" in weights:
            load_fixing_names(model, weights["state_dict"])
        elif "model_state_dict" in weights:
            load_fixing_names(model, weights["model_state_dict"])
        else:
            load_fixing_names(model, weights)
    else:
        indices = inds_file and Indices(inds_file)
        if model_name != "resnet50":
            raise ValueError(f"{model_name} not supported")
        models = {"resnet18": (BasicBlockB, [2, 2, 2, 2]),
                  "resnet34": (BasicBlockB, [3, 4, 6, 3]),
                  "resnet50": (Bottleneck, [3, 4, 6, 3]),
                  "resnet101": (Bottleneck, [3, 4, 23, 3]),
                  "resnet152": (Bottleneck, [3, 8, 36, 3])}
        block, layers = models[model_name]
        model = ResNetB(block, layers, indices=indices)
        if weights_file:
            weights = torch.load(weights_file, map_location="cpu")
            if "state_dict" in weights:
                load_fixing_names(model, weights["state_dict"])
            elif "model_state_dict" in weights:
                load_fixing_names(model, weights["model_state_dict"])
            else:
                load_fixing_names(model, weights)
        else:
            if getattr(torchvision.models, "util", None):
                func = torchvision.models.util.load_state_dict_from_url
            else:
                func = torchvision._internally_replaced_utils.load_state_dict_from_url
            print(f"Loading pretrained model {model_name}")
            state_dict = func(model_urls[model_name], progress=True)
            load_fixing_names(model, state_dict)
    if inds_file:
        model._init_final_weights()
        model.forward = model.forward_decomposed
    return model
