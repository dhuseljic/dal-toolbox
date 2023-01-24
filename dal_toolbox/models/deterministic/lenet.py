import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, in_features=3, num_classes=10):
        super(LeNet, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )

        self.linear_block = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
        )
        self.last = nn.Linear(84, num_classes)

    def forward(self, x, return_features=False):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.linear_block(x)
        features = x.detach()
        out = self.last(x)
        if return_features:
            out = (out, features)
        return out

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
