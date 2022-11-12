import copy
import math
import torch
import torch.nn as nn


class RandomFourierFeatures(nn.Module):
    def __init__(self, in_features, num_inducing=1024, kernel_scale=1, scale_features=True, random_feature_type='orf'):
        super().__init__()

        self.kernel_scale = kernel_scale
        self.input_scale = 1 / math.sqrt(self.kernel_scale)

        self.scale_features = scale_features
        self.random_feature_scale = math.sqrt(2./float(num_inducing))

        self.random_feature_linear = nn.Linear(in_features, num_inducing)
        self.random_feature_linear.weight.requires_grad = False
        self.random_feature_linear.bias.requires_grad = False
        self.reset_parameters(random_feature_type=random_feature_type)

    def reset_parameters(self, random_feature_type='orf'):
        # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/resnet50_sngp.py#L55
        # nn.init.normal_(self.random_feature_linear.weight, std=std_init)
        if random_feature_type == 'rff':
            nn.init.normal_(self.random_feature_linear.weight)
        elif random_feature_type == 'orf':
            orthogonal_random_(self.random_feature_linear.weight)
        else:
            raise ValueError('Only Random Fourier Features `rff` and Orthogonal Random Features `orf` are supported.')

        nn.init.uniform_(self.random_feature_linear.bias, 0, 2*math.pi)

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
                 random_feature_type: str = 'orf',
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
            random_feature_type=random_feature_type,
        )

        # Define output layer according to Eq 8. For imagenet init with normal std=0.01?
        self.beta = nn.Linear(num_inducing, out_features, bias=False)

        # precision matrix
        self.init_precision_matrix = torch.eye(num_inducing)*self.ridge_penalty
        self.register_buffer("precision_matrix", copy.deepcopy(self.init_precision_matrix))
        self.recompute_covariance = True
        self.covariance_matrix = None

    def forward(self, features, return_cov=False):
        if self.normalize_input:
            features = self.layer_norm(features)

        phi = self.random_features(features)

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
        self.covariance_matrix = None

    @torch.no_grad()
    def update_precision_matrix(self, phi, logits):
        # TODO: check multiplier
        # probas = logits.softmax(-1)
        # probas_max = probas.max(1)[0]
        # multiplier = probas_max * (1-probas_max)
        multiplier = 1
        precision_matrix_minibatch = torch.matmul(multiplier*phi.T, phi)
        if self.cov_momentum > 0:
            batch_size = len(phi)
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (self.cov_momentum * self.precision_matrix.data +
                                    (1-self.cov_momentum) * precision_matrix_minibatch)
        else:
            precision_matrix_new = self.precision_matrix.data + precision_matrix_minibatch
        self.precision_matrix.data = precision_matrix_new
        # If there is a change in the precision matrix, recompute the covariance
        self.recompute_covariance = True

    def compute_predictive_covariance(self, phi):
        if self.recompute_covariance:
            # self.cov_mat = torch.linalg.inv(self.precision_matrix.data)
            u = torch.linalg.cholesky(self.precision_matrix.data)
            self.covariance_matrix = torch.cholesky_inverse(u)
        covariance_matrix_feature = self.covariance_matrix.data
        out = torch.matmul(covariance_matrix_feature, phi.T) * self.ridge_penalty
        covariance_matrix_gp = torch.matmul(phi, out)
        return covariance_matrix_gp

    @torch.no_grad()
    def forward_mean_field(self, x):
        if self.training:
            raise ValueError("Call eval mode before!")
        logits, cov = self.forward(x, return_cov=True)
        scaled_logits = mean_field_logits(logits, cov, self.mean_field_factor)
        return scaled_logits


def orthogonal_random_(tensor):
    def sample_ortho(shape):
        return torch.linalg.qr(torch.randn(*shape))[0]

    num_rows, num_cols = tensor.shape
    if num_rows > num_cols:
        ortho_mat_list = []
        num_rows_sampled = 0
        while num_rows_sampled < num_rows:
            ortho_mat_square = sample_ortho((num_cols, num_cols))
            ortho_mat_list.append(ortho_mat_square)
            num_rows_sampled += num_cols
        ortho_mat = torch.cat(ortho_mat_list, dim=0)
        ortho_mat = ortho_mat[:num_rows]
    else:
        ortho_mat = sample_ortho((num_rows, num_cols))

    feature_norms_square = torch.randn(ortho_mat.shape)**2
    feature_norms = torch.sum(feature_norms_square, dim=1).sqrt()
    ortho_mat = feature_norms.unsqueeze(-1) * ortho_mat
    tensor.data = ortho_mat


def mean_field_logits(logits, cov, lmb=math.pi / 8):
    """Scale logits using the mean field approximation proposed by https://arxiv.org/abs/2006.07584"""
    if lmb is None or lmb < 0:
        return logits
    variances = torch.diag(cov).view(-1, 1) if cov is not None else 1
    logits_adjusted = logits / torch.sqrt(1 + lmb*variances)
    return logits_adjusted
