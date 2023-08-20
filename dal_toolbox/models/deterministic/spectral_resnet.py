import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.spectral_normalization import SpectralConv2d, SpectralLinear
from ..utils.random_features import RandomFeatureGaussianProcess


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, spectral_norm=True, norm_bound=1, n_power_iterations=1):
        super(BasicBlock, self).__init__()
        self.conv1 = SpectralConv2d(in_planes, planes, kernel_size=3, spectral_norm=spectral_norm, norm_bound=norm_bound,
                                    n_power_iterations=n_power_iterations, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SpectralConv2d(planes, planes, kernel_size=3, spectral_norm=spectral_norm, norm_bound=norm_bound,
                                    n_power_iterations=n_power_iterations, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SpectralConv2d(in_planes, self.expansion*planes, kernel_size=1, spectral_norm=spectral_norm, norm_bound=norm_bound,
                               n_power_iterations=n_power_iterations, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def spectral_resnet18(num_classes,
                  input_shape,
                  norm_bound,
                  n_power_iterations=1,
                  spectral_norm=True,
                  ):
    resnet18_blocks = [2, 2, 2, 2]
    return SpectralResNet(
        num_classes=num_classes,
        input_shape=input_shape,
        block=BasicBlock,
        num_blocks=resnet18_blocks,
        spectral_norm=spectral_norm,
        norm_bound=norm_bound,
        n_power_iterations=n_power_iterations,
    )


class SpectralResNet(nn.Module):
    def __init__(self,
                 num_classes,
                 input_shape,
                 block,
                 num_blocks,
                 spectral_norm=True,
                 norm_bound=1,
                 n_power_iterations=1,
                 ):
        super(SpectralResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.norm_bound = norm_bound
        self.n_power_iterations = n_power_iterations
        self.spectral_norm = spectral_norm

        # Init layer does not have a kernel size of 7 since cifar has a smaller size of 32x32
        self.conv1 = SpectralConv2d(3, 64, kernel_size=3, spectral_norm=self.spectral_norm, norm_bound=self.norm_bound,
                                    n_power_iterations=self.n_power_iterations, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = SpectralLinear(512 * block.expansion, self.num_classes)

        self.init_spectral_norm(input_shape=self.input_shape)

    @torch.no_grad()
    def init_spectral_norm(self, input_shape):
        # Currently needed for conv layers
        dummy_input = torch.randn((1, *input_shape))
        self(dummy_input)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, self.spectral_norm, self.norm_bound, self.n_power_iterations)
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        features = F.avg_pool2d(out, 4)
        return features.squeeze()

    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []
        for batch in dataloader:
            inputs = batch[0]
            logits = self(inputs.to(device), mean_field=True)
            all_logits.append(logits)
        logits = torch.cat(all_logits)
        return logits

    @torch.inference_mode()
    def get_representations(self, dataloader, device):
        self.to(device)
        self.eval()
        all_features = []
        for batch in dataloader:
            inputs = batch[0]
            logits = self.forward_feature(inputs.to(device))
            all_features.append(logits)
        features = torch.cat(all_features)
        return features
