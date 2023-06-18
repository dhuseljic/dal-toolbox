import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self, in_dimension, num_classes):
        super().__init__()

        self.in_dimension = in_dimension
        self.num_classes = num_classes

        self.linear = nn.Linear(self.in_dimension, self.num_classes)

    def forward(self, x, return_features=False):
        out = self.linear(x)
        if return_features:
            out = (out, x)
        return out

    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []
        for batch in dataloader:
            inputs = batch[0]
            logits = self(inputs.to(device))
            all_logits.append(logits)
        logits = torch.cat(all_logits)
        return logits

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        logits = self.get_logits(dataloader=dataloader, device=device)
        probas = logits.softmax(-1)
        return probas

    @torch.inference_mode()
    def get_representations(self, dataloader, device, return_labels=False):
        all_features = []
        all_labels = []
        for batch in dataloader:
            features = batch[0]
            labels = batch[1]
            all_features.append(features.cpu())
            all_labels.append(labels)
        features = torch.cat(all_features)

        if return_labels:
            labels = torch.cat(all_labels)
            return features, labels
        return features

    @torch.inference_mode()
    def get_grad_representations(self, dataloader, device):
        raise NotImplementedError("Gradient representations are not possible for linear model.")

