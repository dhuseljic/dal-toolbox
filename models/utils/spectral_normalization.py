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


class SpectralNormConv2d(_SpectralNorm):
    def __init__(self,
                 weight: torch.Tensor,
                 norm_bound: float = 1,
                 n_power_iterations: int = 1,
                 dim: int = 0,
                 eps: float = 1e-12):
        super().__init__(weight, n_power_iterations, dim, eps)
        self.norm_bound = norm_bound



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
        if spectral_norm:
            spectral_norm_linear(self, norm_bound=self.norm_bound, n_power_iterations=self.n_power_iterations)
