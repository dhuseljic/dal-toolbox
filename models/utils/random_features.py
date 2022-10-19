import copy
import math
import torch
import torch.nn as nn


class RandomFourierFeatures(nn.Module):
    def __init__(self, in_features, num_inducing=1024, kernel_scale=1, scale_features=True):
        super().__init__()
        self.kernel_scale = kernel_scale
        self.input_scale = 1 / math.sqrt(self.kernel_scale)

        self.scale_features = scale_features
        self.random_feature_scale = math.sqrt(2./float(num_inducing))

        self.random_feature_linear = nn.Linear(in_features, num_inducing)
        self.random_feature_linear.weight.requires_grad = False
        self.random_feature_linear.bias.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self, std_init=1):
        # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/resnet50_sngp.py#L55
        nn.init.normal_(self.random_feature_linear.weight, std=std_init)
        nn.init.uniform_(self.random_feature_linear.bias, 0, 2*math.pi)

        # TODO: Maybe include ORF features as done in
        # https://github.com/y0ast/DUE/blob/f29c990811fd6a8e76215f17049e6952ef5ea0c9/due/sngp.py#L11

    def forward(self, x):
        # Supports lengthscale for cutom random feature layer by directly rescaling the input.
        x = x * self.input_scale
        x = torch.cos(self.random_feature_linear(x))

        # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/wide_resnet_sngp.py#L207
        if self.scale_features:
            # Scale random feature by 2. / sqrt(num_inducing).  When using GP
            # layer as the output layer of a nerual network, it is recommended
            # to turn this scaling off to prevent it from changing the learning
            # rate to the hidden layers.
            x = self.random_feature_scale * x
        return x


class RandomFeatureGaussianProcess(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_inducing: int = 1024,
                 kernel_scale: float = 1,
                 normalize_input: bool = False,
                 scale_random_features: bool = True,
                 mean_field_factor: float = math.pi/8,
                 cov_momentum: float = -1,
                 ridge_penalty: float = 1,
                 ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_inducing = num_inducing

        # scale inputs
        self.kernel_scale = kernel_scale
        self.normalize_input = normalize_input
        self.layer_norm = nn.LayerNorm(in_features)

        # Random features
        self.scale_random_features = scale_random_features

        # Inference
        self.mean_field_factor = mean_field_factor

        # Covariance computation
        self.ridge_penalty = ridge_penalty
        self.cov_momentum = cov_momentum

        self.random_features = RandomFourierFeatures(
            in_features=self.in_features,
            num_inducing=self.num_inducing,
            kernel_scale=self.kernel_scale,
            scale_features=self.scale_random_features,
        )

        # Define output layer according to Eq 8. For imagenet init with normal std=0.01?
        self.beta = nn.Linear(num_inducing, out_features, bias=False)

        # precision matrix
        self.init_precision_matrix = torch.eye(num_inducing)*self.ridge_penalty
        self.register_buffer("precision_matrix", copy.deepcopy(self.init_precision_matrix))
        self.cov_mat = None

    def forward(self, features, return_cov=False):
        if self.normalize_input:
            features = self.layer_norm(features)

        phi = self.random_features(features)

        # Eq. 8
        logits = self.beta(phi)

        if self.training:
            self.update_precision_matrix(phi, logits)
        if return_cov:
            cov = self.compute_predictive_covariance(phi)
            return logits, cov
        return logits

    def reset_precision_matrix(self):
        device = self.precision_matrix.device
        self.precision_matrix.data = copy.deepcopy(self.init_precision_matrix)
        self.precision_matrix.to(device)
        self.cov_mat = None

    @torch.no_grad()
    def update_precision_matrix(self, phi, logits):
        probas = logits.softmax(-1)
        probas_max = probas.max(1)[0]
        multiplier = probas_max * (1-probas_max)
        precision_matrix_minibatch = torch.matmul(
            multiplier*phi.T, phi
        )
        if self.cov_momentum > 0:
            batch_size = len(phi)
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (self.cov_momentum * self.precision_matrix.data +
                                    (1-self.cov_momentum) * precision_matrix_minibatch)
        else:
            precision_matrix_new = self.precision_matrix.data + precision_matrix_minibatch
        self.precision_matrix.data = precision_matrix_new
        self.cov_mat = None

    @property
    def covariance_matrix(self):
        device = self.precision_matrix.data.device
        if self.cov_mat is None:
            self.cov_mat = torch.linalg.inv(self.precision_matrix.data)
        return self.cov_mat.to(device)

    @torch.no_grad()
    def forward_mean_field(self, x):
        if self.training:
            raise ValueError("Call eval mode before!")
        logits, cov = self.forward(x, return_cov=True)
        scaled_logits = mean_field_logits(logits, cov, self.mean_field_factor)
        return scaled_logits

    def compute_predictive_covariance(self, phi):
        covariance_matrix_feature = self.covariance_matrix.data
        out = torch.matmul(covariance_matrix_feature, phi.T) * self.ridge_penalty
        covariance_matrix_gp = torch.matmul(phi, out)
        return covariance_matrix_gp


def mean_field_logits(logits, cov, lmb=math.pi / 8):
    """Scale logits using the mean field approximation proposed by https://arxiv.org/abs/2006.07584"""
    if lmb is None or lmb < 0:
        return logits
    variances = torch.diag(cov).view(-1, 1) if cov is not None else 1
    logits_adjusted = logits / torch.sqrt(1 + lmb*variances)
    return logits_adjusted
