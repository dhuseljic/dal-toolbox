import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.mcdropout import MCDropoutModule, ConsistentMCDropout2d


class DropoutResNet18(MCDropoutModule):
    def __init__(self, num_classes=10, n_passes=10, dropout_rate=0.2):
        super().__init__(n_passes=n_passes)
        self.in_planes = 64
        self.block = DropoutBasicBlock
        self.num_blocks = [2, 2, 2, 2]
        self.n_passes = n_passes
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_dropout = ConsistentMCDropout2d(self.dropout_rate)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], self.dropout_rate, stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], self.dropout_rate, stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], self.dropout_rate, stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], self.dropout_rate, stride=2)
        self.linear = nn.Linear(512*self.block.expansion, num_classes)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    def _make_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, get_embeddings=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv1_dropout(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = (self.linear(out), out) if get_embeddings else self.linear(out)
        return out

    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        self.to(device)
        self.eval()
        mc_logits_list = []
        for batch in dataloader:
            samples = batch[0]
            mc_logits = self.mc_forward(samples.to(device))
            mc_logits_list.append(mc_logits.cpu())
        mc_logits = torch.cat(mc_logits_list)
        return mc_logits


class DropoutBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1_dropout = ConsistentMCDropout2d(dropout_rate)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2_dropout = ConsistentMCDropout2d(dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv1_dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        # out = self.conv2_dropout(out)
        return out
