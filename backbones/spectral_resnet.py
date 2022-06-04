# TODO: Adapted implementation from ...
import torch.nn as nn
import torch.nn.functional as F
from .spectral_norm import SpectralConv2d


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, spectral_norm=True, stride=1, norm_bound=1, n_power_iterations=1):
        super(Bottleneck, self).__init__()
        self.conv1 = SpectralConv2d(in_planes, planes, kernel_size=1, spectral_norm=spectral_norm, norm_bound=norm_bound,
                                    n_power_iterations=n_power_iterations, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SpectralConv2d(planes, planes, kernel_size=3, spectral_norm=spectral_norm, norm_bound=norm_bound,
                                    n_power_iterations=n_power_iterations, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SpectralConv2d(planes, self.expansion * planes, kernel_size=1, spectral_norm=spectral_norm,
                                    norm_bound=norm_bound, n_power_iterations=n_power_iterations, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SpectralConv2d(in_planes, self.expansion*planes, kernel_size=1, spectral_norm=spectral_norm, norm_bound=norm_bound,
                               n_power_iterations=n_power_iterations, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, spectral_norm=True, norm_bound=1, n_power_iterations=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm_bound = norm_bound
        self.n_power_iterations = n_power_iterations
        self.spectral_norm = spectral_norm

        # Init layer does not have a kernel size of 7 since cifar has a smaller
        # size of 32x32
        self.conv1 = SpectralConv2d(3, 64, kernel_size=3, spectral_norm=self.spectral_norm, norm_bound=self.norm_bound,
                                    n_power_iterations=self.n_power_iterations, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, self.spectral_norm, self.norm_bound, self.n_power_iterations)
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        self.features = out
        out = self.linear(out)
        if return_features:
            out = (out, self.features)
        return out


def spectral_resnet18(num_classes, norm_bound, n_power_iterations=1, spectral_norm=True):

    return ResNet(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        spectral_norm=spectral_norm,
        norm_bound=norm_bound,
        n_power_iterations=n_power_iterations,
        num_classes=num_classes
    )


def spectral_resnet34(num_classes, norm_bound, n_power_iterations=1, spectral_norm=True):
    return ResNet(
        block=BasicBlock,
        num_blocks=[3, 4, 6, 3],
        spectral_norm=spectral_norm,
        norm_bound=norm_bound,
        n_power_iterations=n_power_iterations,
        num_classes=num_classes
    )


def spectral_resnet50(num_classes, norm_bound, n_power_iterations=1, spectral_norm=True):
    return ResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 6, 3],
        spectral_norm=spectral_norm,
        norm_bound=norm_bound,
        n_power_iterations=n_power_iterations,
        num_classes=num_classes
    )


def spectral_resnet101(num_classes, norm_bound, n_power_iterations=1, spectral_norm=True):
    return ResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 23, 3],
        spectral_norm=spectral_norm,
        norm_bound=norm_bound,
        n_power_iterations=n_power_iterations,
        num_classes=num_classes
    )


def spectral_resnet152(num_classes, norm_bound, n_power_iterations=1, spectral_norm=True):
    return ResNet(
        block=Bottleneck,
        num_blocks=[3, 8, 36, 3],
        spectral_norm=spectral_norm,
        norm_bound=norm_bound,
        n_power_iterations=n_power_iterations,
        num_classes=num_classes
    )


# model = spectral_resnet18(num_classes=10, norm_bound=6, n_power_iterations=1, spectral_norm=True)
# for m in model.modules():
#     if isinstance(m, nn.Conv2d):
#         print(m.weight_u)
#         print(m.norm_bound, m.n_power_iterations)