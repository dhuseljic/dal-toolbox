import numpy as np
from torch import norm

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .spectral_norm import SpectralConv2d


def conv3x3(in_planes, out_planes, stride=1, spectral_norm=True, norm_bound=1, n_power_iterations=1):
    return SpectralConv2d(in_planes, out_planes, kernel_size=3, spectral_norm=spectral_norm, norm_bound=norm_bound,
                          n_power_iterations=n_power_iterations, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, spectral_norm=True, norm_bound=1, n_power_iterations=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = SpectralConv2d(in_planes, planes, kernel_size=3, spectral_norm=spectral_norm,
                                    norm_bound=norm_bound, n_power_iterations=n_power_iterations, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = SpectralConv2d(planes, planes, kernel_size=3, spectral_norm=spectral_norm,
                                    norm_bound=norm_bound, n_power_iterations=n_power_iterations, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                SpectralConv2d(in_planes, planes, kernel_size=1, spectral_norm=spectral_norm,
                               norm_bound=norm_bound, n_power_iterations=n_power_iterations, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, spectral_norm, norm_bound, n_power_iterations):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        # Spectral norm params
        self.spectral_norm = spectral_norm
        self.norm_bound = norm_bound
        self.n_power_iterations = n_power_iterations

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0], spectral_norm=self.spectral_norm,
                             norm_bound=self.norm_bound, n_power_iterations=self.n_power_iterations)
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, dropout_rate=dropout_rate, stride=stride,
                          spectral_norm=self.spectral_norm, norm_bound=self.norm_bound, n_power_iterations=self.n_power_iterations))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        self.features = out
        out = self.linear(out)
        if return_features:
            out = (out, self.features)

        return out


# model = WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10,
#                    spectral_norm=True, norm_bound=6, n_power_iterations=1)
# for m in model.modules():
#     if isinstance(m, nn.Conv2d):
#         print(m.weight_u)
#         print(m.norm_bound, m.n_power_iterations)