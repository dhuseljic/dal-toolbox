import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class BayesianModule(nn.Module):
    """A module that we can sample multiple times from given a single input batch.

    To be efficient, the module allows for a part of the forward pass to be deterministic.
    """

    k = None

    def __init__(self):
        super().__init__()

    # Returns B x n x output
    def mc_forward(self, input_B: torch.Tensor, k: int):
        BayesianModule.k = k
        mc_input_BK = BayesianModule.mc_tensor(input_B, k)
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = BayesianModule.unflatten_tensor(mc_output_BK, k)
        return mc_output_B_K

    def mc_forward_impl(self, mc_input_BK: torch.Tensor):
        return mc_input_BK

    def get_mc_logits(self, dataloader, k, device):
        mc_probas = []
        self.to(device)
        self.eval()
        for samples, _ in tqdm(dataloader):
            samples = samples.to(device)
            mc_probas.append(self.mc_forward(samples, k))
        return torch.cat(mc_probas)


    @staticmethod
    def unflatten_tensor(input: torch.Tensor, k: int):
        input = input.view([-1, k] + list(input.shape[1:]))
        return input

    @staticmethod
    def flatten_tensor(mc_input: torch.Tensor):
        return mc_input.flatten(0, 1)

    @staticmethod
    def mc_tensor(input: torch.tensor, k: int):
        mc_shape = [input.shape[0], k] + list(input.shape[1:])
        return input.unsqueeze(1).expand(mc_shape).flatten(0, 1)


class _ConsistentMCDropout(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))

        self.p = p
        self.mask = None

    def extra_repr(self):
        return "p={}".format(self.p)

    def reset_mask(self):
        self.mask = None

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self.reset_mask()

    def _get_sample_mask_shape(self, sample_shape):
        return sample_shape

    def _create_mask(self, input, k):
        mask_shape = [1, k] + list(self._get_sample_mask_shape(input.shape[1:]))
        mask = torch.empty(mask_shape, dtype=torch.bool, device=input.device).bernoulli_(self.p)
        return mask

    def forward(self, input: torch.Tensor):
        if self.p == 0.0:
            return input

        k = BayesianModule.k
        if self.training:
            # Create a new mask on each call and for each batch element.
            k = input.shape[0]
            mask = self._create_mask(input, k)
        else:
            if self.mask is None:
                # print('recreating mask', self)
                # Recreate mask.
                self.mask = self._create_mask(input, k)

            mask = self.mask

        mc_input = BayesianModule.unflatten_tensor(input, k)
        mc_output = mc_input.masked_fill(mask, 0) / (1 - self.p)

        # Flatten MCDI, batch into one dimension again.
        return BayesianModule.flatten_tensor(mc_output)


class ConsistentMCDropout2d(_ConsistentMCDropout):
    def _get_sample_mask_shape(self, sample_shape):
        return [sample_shape[0]] + [1] * (len(sample_shape) - 1)



class DropoutResNet(BayesianModule):
    def __init__(self, block, num_blocks, num_classes=10, p_drop=0.2):
        super(BayesianModule, self).__init__()
        self.in_planes = 64
        self.dropout = True

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv1_drop = ConsistentMCDropout2d(p_drop)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0] , p_drop, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1] , p_drop, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2] , p_drop, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3] , p_drop, stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    def _make_layer(self, block, planes, num_blocks, p_drop, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, p_drop, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, get_embeddings=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = (self.linear(out), out) if get_embeddings else self.linear(out)
        return out 

    def mc_forward_impl(self, mc_input_BK: torch.Tensor):
        out = self.conv1(mc_input_BK)
        out = self.conv1_drop(out)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def set_dropout(self, mod):
        self.dropout=mod
        for seq in self.layers:
            for module in seq:
                module.dropout=mod


class DropoutBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, p_drop, stride=1):
        super().__init__()
        self.dropout = True

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1_drop = ConsistentMCDropout2d(p_drop)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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
        out = self.conv1_drop(out) if self.dropout else out
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DropoutBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, p_drop, stride=1):
        super().__init__()
        self.dropout = True

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv1_drop = ConsistentMCDropout2d(p_drop)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2_drop = ConsistentMCDropout2d(p_drop)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_drop(out) if self.dropout else out
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.conv2_drop(out) if self.dropout else out
        out = F.relu(self.bn2(out))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out