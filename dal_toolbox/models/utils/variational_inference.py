# Inspired by: https://github.com/Harry24k/bayesian-neural-network-pytorch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Implemente that a bias term can be ignored
# TODO: Implement Bayesian batch norm


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, prior_sigma=1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.prior_sigma = prior_sigma

        # Weight mean and variance
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            # Bias mean and variance
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # init from torchbnn
        fan_in = self.weight_mu.size(1)
        stdv = 1. / math.sqrt(fan_in)
        nn.init.uniform_(self.weight_mu, -stdv, stdv)
        # nn.init.kaiming_uniform_(self.weight_mu)
        nn.init.normal_(self.weight_rho, -5, 0.01)

        if self.bias:
            # nn.init.zeros_(self.bias_mu)
            nn.init.uniform_(self.bias_mu, -stdv, stdv)
            nn.init.normal_(self.bias_rho, -5, 0.01)

    def forward(self, x):
        # Sample the weights and biases using the reparameterization trick
        weight_epsilon = torch.randn_like(self.weight_mu)
        weight_sigma = F.softplus(self.weight_rho)
        weight_sample = self.weight_mu + weight_sigma * weight_epsilon

        if self.bias:
            bias_epsilon = torch.randn_like(self.bias_mu)
            bias_sigma = F.softplus(self.bias_rho)
            bias_sample = self.bias_mu + bias_sigma * bias_epsilon
        else:
            bias_sample = None

        # Perform the linear operation using F.linear
        return F.linear(x, weight_sample, bias_sample)

    def kl_divergence(self):
        kl_divergence = 0

        weight_sigma = F.softplus(self.weight_rho)
        kl_weight = 0.5 * torch.sum(((self.weight_mu**2 + weight_sigma**2) / self.prior_sigma**2) +
                                    2 * (math.log(self.prior_sigma) - torch.log(weight_sigma)) - 1)
        kl_divergence = kl_divergence + kl_weight

        if self.bias:
            bias_sigma = F.softplus(self.bias_rho)
            kl_bias = 0.5 * torch.sum(((self.bias_mu**2 + bias_sigma**2) / self.prior_sigma**2) +
                                      2 * (math.log(self.prior_sigma) - torch.log(bias_sigma)) - 1)
            kl_divergence = kl_divergence + kl_bias

        return kl_divergence


class BayesianConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 prior_sigma: float = 1.0):
        super(BayesianConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.prior_sigma = prior_sigma

        # Weight mean and variance
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))

        if bias:
            # Bias mean and variance
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        # init from torchbnn
        fan_in = self.in_channels * self.kernel_size[0] ** 2
        stdv = 1.0 / math.sqrt(fan_in)

        # nn.init.kaiming_uniform_(self.weight_mu)
        nn.init.uniform_(self.weight_mu, -stdv, stdv)
        nn.init.normal_(self.weight_rho, -5, 0.01)

        if self.bias:
            # nn.init.zeros_(self.bias_mu)
            nn.init.uniform_(self.bias_mu, -stdv, stdv)
            nn.init.normal_(self.bias_rho, -5, 0.01)

    def forward(self, x):
        # Sample the weights and biases using the reparameterization trick
        weight_epsilon = torch.randn_like(self.weight_mu)
        weight_sigma = F.softplus(self.weight_rho)
        weight_sample = self.weight_mu + weight_sigma * weight_epsilon

        if self.bias:
            bias_epsilon = torch.randn_like(self.bias_mu)
            bias_sigma = F.softplus(self.bias_rho)
            bias_sample = self.bias_mu + bias_sigma * bias_epsilon
        else:
            bias_sample = None

        return F.conv2d(x, weight_sample, bias_sample, self.stride, self.padding, self.dilation, self.groups)

    def kl_divergence(self):
        kl_divergence = 0

        weight_sigma = F.softplus(self.weight_rho)
        kl_weight = 0.5 * torch.sum(((self.weight_mu**2 + weight_sigma**2) / self.prior_sigma**2) +
                                    2 * (math.log(self.prior_sigma) - torch.log(weight_sigma)) - 1)
        kl_divergence = kl_divergence + kl_weight

        if self.bias:
            bias_sigma = F.softplus(self.bias_rho)
            kl_bias = 0.5 * torch.sum(((self.bias_mu**2 + bias_sigma**2) / self.prior_sigma**2) +
                                      2 * (math.log(self.prior_sigma) - torch.log(bias_sigma)) - 1)
            kl_divergence = kl_divergence + kl_bias

        return kl_divergence


def kl_criterion(model, reduction='mean'):
    num_parameters = 0
    kl_divergence = 0

    for module in model.modules():
        if isinstance(module, (BayesianLinear, BayesianConv2d)):
            kl_divergence += module.kl_divergence()
            num_parameters += module.weight_mu.numel()
            if module.bias:
                num_parameters += module.bias_mu.numel()

    if reduction == 'mean':
        return kl_divergence/num_parameters
    elif reduction == 'sum':
        return kl_divergence
    else:
        raise NotImplementedError(f'Reduction type not implemented, got {reduction}')


class KLCriterion(nn.Module):
    def __init__(self, reduction='mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, model):
        return kl_criterion(model, reduction='mean')
