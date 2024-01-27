import copy
import math
import torch
import torch.nn as nn
import torch.distributed as dist

from dal_toolbox.utils import is_dist_avail_and_initialized
from .random_features import mean_field_logits


class LaplaceLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 mean_field_factor: float = math.pi/8,
                 mc_samples: int = 1000,
                 cov_momentum: float = -1,
                 ridge_penalty: float = 1,
                 bias: bool = True,
                 ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.mean_field_factor = mean_field_factor
        self.mc_samples = mc_samples

        self.cov_momentum = cov_momentum
        self.ridge_penalty = ridge_penalty

        self.layer = nn.Linear(in_features, out_features, bias=bias)

        self.init_precision_matrix = torch.eye(in_features)*self.ridge_penalty
        self.register_buffer("precision_matrix", copy.deepcopy(self.init_precision_matrix))
        self.register_buffer("covariance_matrix", torch.full(self.precision_matrix.shape, float('nan')))
        self.recompute_covariance = True

    def forward(self, inputs, return_cov=False, return_features=False):
        logits = self.layer(inputs)

        if self.training:
            self.update_precision_matrix(inputs, logits)

        if return_cov:
            cov = self.compute_predictive_covariance(inputs)
            return logits, cov

        if return_features:
            return logits, inputs

        return logits

    def reset_precision_matrix(self):
        device = self.precision_matrix.device
        self.precision_matrix.data = copy.deepcopy(self.init_precision_matrix)
        self.precision_matrix.to(device)
        self.recompute_covariance = True
        self.covariance_matrix.data = torch.full(self.precision_matrix.shape, float('nan'))

    def synchronize_precision_matrix(self):
        if not is_dist_avail_and_initialized():
            return
        # Sum all precision_matrices
        dist.all_reduce(self.precision_matrix, op=dist.ReduceOp.SUM)
        # The init precision matrix is summed world_size times. However, it
        # should be only one time. Thus we need to subtract the
        # init_precision_matrix (1 - world_size)-times
        init_precision_matrix = self.init_precision_matrix.to(self.precision_matrix)
        self.precision_matrix = self.precision_matrix - (dist.get_world_size()-1)*init_precision_matrix

    @torch.no_grad()
    def update_precision_matrix(self, inputs, logits):
        # probas = logits.softmax(-1)
        # probas_max = probas.max(1)[0]
        # multiplier = probas_max * (1-probas_max)
        # we assume a gaussian likelihood like google
        multiplier = 1
        self.precision_matrix = self.precision_matrix.to(inputs)
        precision_matrix_minibatch = torch.matmul(multiplier*inputs.T, inputs)
        if self.cov_momentum > 0:
            batch_size = len(inputs)
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (self.cov_momentum * self.precision_matrix.data +
                                    (1-self.cov_momentum) * precision_matrix_minibatch)
        else:
            precision_matrix_new = self.precision_matrix.data + precision_matrix_minibatch
        self.precision_matrix.data = precision_matrix_new
        # If there is a change in the precision matrix, recompute the covariance
        self.recompute_covariance = True

    def compute_predictive_covariance(self, inputs):
        if self.recompute_covariance:
            # self.covariance_matrix  = torch.linalg.inv(self.precision_matrix.data)
            u = torch.linalg.cholesky(self.precision_matrix.data)
            self.covariance_matrix.data = torch.cholesky_inverse(u)
        covariance_matrix_feature = self.covariance_matrix.data
        out = torch.matmul(covariance_matrix_feature, inputs.T) * self.ridge_penalty
        covariance_matrix_gp = torch.matmul(inputs, out)
        return covariance_matrix_gp

    @torch.no_grad()
    def forward_mean_field(self, x):
        if self.training:
            raise ValueError("Call eval mode before!")
        logits, cov = self.forward(x, return_cov=True)
        scaled_logits = mean_field_logits(logits, cov, self.mean_field_factor)
        return scaled_logits

    @torch.no_grad()
    def forward_monte_carlo(self, x):
        if self.training:
            raise ValueError("Call eval mode before!")
        logits, cov = self.forward(x, return_cov=True)

        gp_mean = logits
        gp_var = cov.diag()
        gp_var = gp_var.unsqueeze(-1).expand_as(gp_mean)
        gp_std = torch.sqrt(gp_var)

        gaussian = torch.distributions.Normal(loc=gp_mean,  scale=gp_std)
        mc_logits = gaussian.sample(sample_shape=(self.mc_samples,)).permute(1, 0, 2)
        return mc_logits
