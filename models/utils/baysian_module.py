import torch
import torch.nn as nn
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