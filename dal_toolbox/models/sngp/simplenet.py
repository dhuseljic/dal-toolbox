import torch
import torch.nn as nn
from dal_toolbox.models.utils.random_features import RandomFeatureGaussianProcess
from dal_toolbox.models.utils.spectral_normalization import spectral_norm_linear


class SimpleSNGP(nn.Module):
    def __init__(self,
                 num_classes: int,
                 use_spectral_norm: bool = True,
                 spectral_norm_params: dict = {},
                 gp_params: dict = {},
                 n_residual_layers: int = 6,
                 feature_dim: int = 128,
                 ):
        super().__init__()

        def linear_layer(*args, **kwargs):
            if use_spectral_norm:
                return spectral_norm_linear(nn.Linear(*args, **kwargs), **spectral_norm_params)
            else:
                return nn.Linear(*args, **kwargs)

        self.first = nn.Linear(2, feature_dim)
        self.residuals = nn.ModuleList([linear_layer(128, 128) for _ in range(n_residual_layers)])
        self.last = RandomFeatureGaussianProcess(
            in_features=feature_dim,
            out_features=num_classes,
            **gp_params,
            mc_samples=1000
        )
        self.act = nn.ReLU()
    

    def forward(self, x, mean_field=False, monte_carlo=False, return_cov=False):
        x = self.act(self.first(x))
        for residual in self.residuals:
            x = self.act(residual(x) + x)

        if mean_field:
            out = self.last.forward_mean_field(x)
        elif monte_carlo:
            out = self.last.forward_monte_carlo(x)
        else:
            out = self.last(x, return_cov=return_cov)

        return out
    
    def forward_feature(self, x):
        x = self.act(self.first(x))
        for residual in self.residuals:
            x = self.act(residual(x) + x)
        return x.squeeze()
    
    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []
        for batch in dataloader:
            inputs = batch[0]
            # TODO: Handle with forward_kwargs? currently only monte carlo supported
            logits = self(inputs.to(device), monte_carlo=True)
            all_logits.append(logits)
        logits = torch.cat(all_logits)
        logits = torch.mean(logits, dim=1)
        return logits

    @torch.inference_mode()
    def get_representations(self, dataloader, device):
        self.to(device)
        self.eval()
        all_features = []
        for batch in dataloader:
            inputs = batch[0]
            logits = self.forward_feature(inputs.to(device))
            all_features.append(logits)
        features = torch.cat(all_features)
        return features