import torch
import torch.nn as nn
from tqdm.auto import tqdm


class MCDropoutModule(nn.Module):
    """A module that we can sample multiple times from given a single input batch.

    To be efficient, the module allows for a part of the forward pass to be deterministic.
    """
    n_passes = None

    def __init__(self, n_passes):
        super().__init__()
        MCDropoutModule.n_passes = n_passes

    # Returns B x n x output
    def mc_forward(self, input_B: torch.Tensor):
        mc_input_BK = MCDropoutModule.mc_tensor(input_B, MCDropoutModule.n_passes)
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = MCDropoutModule.unflatten_tensor(mc_output_BK, MCDropoutModule.n_passes)
        return mc_output_B_K  # N x M x K

    def mc_forward_impl(self, *args, **kwargs):
        return self(*args, **kwargs)

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

        if self.training:
            # Create a new mask on each call and for each batch element.
            mask = self._create_mask(input, input.shape[0])
        else:
            if self.mask is None:
                # print('recreating mask', self)
                # Recreate mask.
                self.mask = self._create_mask(input, MCDropoutModule.n_passes)

            mask = self.mask

        k = input.shape[0] if self.training else MCDropoutModule.n_passes
        mc_input = MCDropoutModule.unflatten_tensor(input, k)
        mc_output = mc_input.masked_fill(mask, 0) / (1 - self.p)

        # Flatten MCDI, batch into one dimension again.
        return MCDropoutModule.flatten_tensor(mc_output)


class ConsistentMCDropout(_ConsistentMCDropout):
    r"""Randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. The elements to zero are randomized on every forward call during training time.
    During eval time, a fixed mask is picked and kept until `reset_mask()` is called.
    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .
    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    Shape:
        - Input: `Any`. Input can be of any shape
        - Output: `Same`. Output is of the same shape as input
    Examples::
        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)
    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """


class ConsistentMCDropout2d(_ConsistentMCDropout):
    def _get_sample_mask_shape(self, sample_shape):
        return [sample_shape[0]] + [1] * (len(sample_shape) - 1)
