import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.variational_inference import BayesianConv2d, BayesianLinear


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, prior_sigma=1):
        super(BasicBlock, self).__init__()
        self.prior_sigma = prior_sigma
        self.conv1 = BayesianConv2d(in_planes, planes, kernel_size=3, stride=stride,
                                    padding=1, bias=False, prior_sigma=prior_sigma)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BayesianConv2d(planes, planes, kernel_size=3, stride=1,
                                    padding=1, bias=False, prior_sigma=prior_sigma)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BayesianConv2d(in_planes, self.expansion*planes, kernel_size=1,
                               stride=stride, bias=False, prior_sigma=prior_sigma),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BayesianResNet18(nn.Module):
    def __init__(self, num_classes, prior_sigma=1):
        super(BayesianResNet18, self).__init__()
        self.in_planes = 64
        self.block = BasicBlock
        self.num_blocks = [2, 2, 2, 2]
        self.num_classes = num_classes
        self.prior_sigma = prior_sigma

        # Init layer does not have a kernel size of 7 since cifar has a smaller
        # size of 32x32
        self.conv1 = BayesianConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, prior_sigma=prior_sigma)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.linear = BayesianLinear(512*self.block.expansion, self.num_classes, prior_sigma=prior_sigma)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, prior_sigma=self.prior_sigma))
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
        features = out
        out = self.linear(out)
        if return_features:
            out = (out, features)
        return out

    @torch.no_grad()
    def forward_sample(self, x, mc_samples=10):
        mc_logits = []
        for _ in range(mc_samples):
            mc_logits.append(self.forward(x))
        mc_logits = torch.stack(mc_logits, dim=1)
        return mc_logits

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []
        for samples, _ in dataloader:
            logits = self(samples.to(device))
            all_logits.append(logits)
        logits = torch.cat(all_logits)
        probas = logits.softmax(-1)
        return probas

    @torch.inference_mode()
    def get_representation(self, dataloader, device):
        self.to(device)
        self.eval()
        all_features = []
        for samples, _ in dataloader:
            _, features = self(samples.to(device), return_features=True)
            all_features.append(features.cpu())
        features = torch.cat(all_features)
        return features

    @torch.inference_mode()
    def get_grad_embedding(self, dataloader, device):
        self.eval()
        self.to(device)
        feature_dim = 512

        embedding = []
        for inputs, _ in dataloader:
            embedding_batch = torch.empty([len(inputs), feature_dim * self.num_classes])
            logits, features = self(inputs.to(device), return_features=True)
            logits = logits.cpu()
            features = features.cpu()

            probas = logits.softmax(-1)
            max_indices = probas.argmax(-1)

            # TODO: optimize code
            # for each sample in a batch and for each class, compute the gradient wrt to weights
            for n in range(len(inputs)):
                for c in range(self.num_classes):
                    if c == max_indices[n]:
                        embedding_batch[n, feature_dim * c: feature_dim * (c+1)] = features[n] * (1 - probas[n, c])
                    else:
                        embedding_batch[n, feature_dim * c: feature_dim * (c+1)] = features[n] * (-1 * probas[n, c])
            embedding.append(embedding_batch)
        # Concat all embeddings
        embedding = torch.cat(embedding)
        return embedding
