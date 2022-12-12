import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import _SpectralNorm


class SpectralNormLinear(_SpectralNorm):
    def __init__(self,
                 weight: torch.Tensor,
                 norm_bound: float = 1,
                 n_power_iterations: int = 1,
                 dim: int = 0,
                 eps: float = 1e-12):
        super().__init__(weight, n_power_iterations, dim, eps)
        self.norm_bound = norm_bound

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.dot(u, torch.mv(weight_mat, v))

            # slower version: weight_norm = torch.where((self.norm_bound / sigma) < 1, (self.norm_bound / sigma) * weight, weight)

            factor = torch.max(torch.ones(1, device=weight.device), sigma / self.norm_bound)
            weight_norm = weight / factor

            return weight_norm


def spectral_norm_linear(module: Module, norm_bound: float, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    weight = getattr(module, name, None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    parametrize.register_parametrization(
        module,
        name,
        SpectralNormLinear(weight, norm_bound, n_power_iterations, dim, eps)
    )
    return module


class SpectralLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 spectral_norm=True,
                 norm_bound=1,
                 n_power_iterations=1,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        # Apply spectral norm after init
        self.norm_bound = norm_bound
        self.n_power_iterations = n_power_iterations
        if spectral_norm:
            spectral_norm_linear(self, norm_bound=self.norm_bound, n_power_iterations=self.n_power_iterations)


# Adapted from:
# https://github.com/google/edward2/blob/5338ae3244b90f3fdd0cf10094937c09eb40fab9/edward2/tensorflow/layers/normalization.py#L398-L541
# https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/parametrizations.py
class SpectralNormConv2d(nn.Module):
    def __init__(self,
                 weight: torch.Tensor,
                 input_shape: torch.Tensor,
                 output_shape: torch.Tensor,
                 stride: tuple,
                 padding: int,
                 norm_bound: float = 1,
                 n_power_iterations: int = 1,
                 dim: int = 0,
                 eps: float = 1e-12):
        super().__init__()
        self.norm_bound = norm_bound

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.stride = stride
        self.padding = padding

        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError("Dimension out of range (expected to be in range of "
                             f"[-{ndim}, {ndim - 1}] but got {dim})")

        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))

        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        if ndim > 1:
            # For ndim == 1 we do not need to approximate anything (see _SpectralNorm.forward)
            self.n_power_iterations = n_power_iterations

            v = torch.randn(self.input_shape).to(weight.device)
            v_ = F.normalize(v.view(-1), dim=0, eps=self.eps)
            v = v_.view(v.shape)

            u = torch.randn(self.output_shape).to(weight.device)
            u_ = F.normalize(u.view(-1), dim=0, eps=self.eps)
            u = u_.view(u.shape)

            self.register_buffer('_v', v)
            self.register_buffer('_u', u)
            self._power_method(weight, 15)
            # self._u.mean(), self._u.std()

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
        # Does not support Dataparallel, see comment in pytorch implementation
        assert weight_mat.ndim > 1

        u_hat = getattr(self, '_u')
        v_hat = getattr(self, '_v')

        for _ in range(n_power_iterations):
            v_ = F.conv_transpose2d(u_hat, weight_mat, stride=self.stride, padding=self.padding)
            v_hat = F.normalize(v_.view(1, -1))
            v_hat = v_hat.view(v_.shape)

            u_ = F.conv2d(v_hat, weight_mat, stride=self.stride, padding=self.padding)
            u_hat = F.normalize(u_.view(1, -1))
            u_hat = u_hat.view(u_.shape)

        setattr(self, '_u', u_hat)
        setattr(self, '_v', v_hat)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._power_method(weight, self.n_power_iterations)
        u = self._u.clone(memory_format=torch.contiguous_format)
        v = self._v.clone(memory_format=torch.contiguous_format)

        v_w_hat = F.conv2d(v, weight, stride=self.stride, padding=self.padding)
        sigma = torch.matmul(v_w_hat.view(1, -1), u.view(-1, 1))
        sigma = sigma.squeeze()

        factor = torch.max(torch.ones(1, device=weight.device), sigma / self.norm_bound)
        weight_norm = weight / factor

        return weight_norm


def spectral_norm_conv2d(module: Module,
                         input_shape: tuple,
                         output_shape: tuple,
                         stride: tuple,
                         padding: int,
                         norm_bound: float,
                         name='weight',
                         n_power_iterations=1,
                         eps=1e-12,
                         dim=None):
    weight = getattr(module, name, None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError("Module '{}' has no parameter or buffer with name '{}'".format(module, name))
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    parametrize.register_parametrization(
        module,
        name,
        SpectralNormConv2d(weight, input_shape, output_shape, stride, padding, norm_bound, n_power_iterations, dim, eps)
        # SpectralNormLinear(weight, norm_bound, n_power_iterations, dim, eps)
    )
    return module


class SpectralConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 spectral_norm=True,
                 norm_bound=1,
                 n_power_iterations=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype
        )
        # Apply spectral norm after init
        self.norm_bound = norm_bound
        self.n_power_iterations = n_power_iterations
        self.spectral_norm = spectral_norm
        self.spectral_norm_added = False

    def add_spectral_norm_conv2d(self, input):
        if self.spectral_norm:
            # Determine shapes
            in_channel, in_height, in_width = input.shape[1:]

            out_channel = self.out_channels
            out_height = in_height // self.stride[0]
            out_width = in_width // self.stride[1]

            input_shape = (1, in_channel, in_height, in_width)
            output_shape = (1, out_channel, out_height, out_width)

            spectral_norm_conv2d(
                module=self,
                input_shape=input_shape,
                output_shape=output_shape,
                stride=self.stride,
                padding=self.padding,
                norm_bound=self.norm_bound,
                n_power_iterations=self.n_power_iterations,
            )
            self.spectral_norm_added = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super().forward(input)
        if not self.spectral_norm_added:
            self.add_spectral_norm_conv2d(input)
        return out
