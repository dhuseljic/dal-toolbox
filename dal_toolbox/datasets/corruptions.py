import torch

class GaussianNoise(torch.nn.Module):
    def __init__(self, severity):
        super(GaussianNoise, self).__init__()
        self.severity = severity

    def forward(self, x):
        return (1-self.severity)*x + self.severity*torch.rand(*x.shape)