import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self, in_dimension, num_classes):
        super().__init__()

        self.in_dimension = in_dimension
        self.num_classes = num_classes

        self.linear = nn.Linear(self.in_dimension, self.num_classes)
        self.dropout = nn.Dropout()

    def set_dropout(self, p):
        self.dropout = nn.Dropout(p=p)

    def forward(self, x, return_features=False):
        out = self.linear(x)
        if return_features:
            out = (out, x)
        return out
    
    def forward_dropout(self, x, return_features=False):
        x = self.dropout(x)
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
    def get_representations(self, dataloader, return_labels=False, device='cuda'):
        self.to(device)
        all_features = []
        all_labels = []
        for batch in dataloader:
            features = batch[0]
            labels = batch[1]
            all_features.append(features.to(device))
            all_labels.append(labels)
        features = torch.cat(all_features)

        if return_labels:
            labels = torch.cat(all_labels)
            return features, labels
        return features
    
    @torch.inference_mode()
    def get_representations_and_probas(self, dataloader):
        all_features = []
        all_probas = []
        for batch in dataloader:
            features = batch[0]
            all_features.append(features.cpu())
            all_probas.append(self(features).softmax(-1))
        features = torch.cat(all_features)
        probas = torch.cat(all_probas)
        return features, probas


    @torch.inference_mode()
    def get_grad_representations(self, dataloader, device, return_pseudo_labels=False, return_embeddings=False):
        self.eval()
        self.to(device)

        embeddings, gradients, pseudo_labels = [], [], []
        for batch in dataloader:
            inputs = batch[0].to(device)
            logits = self(inputs)

            probas = logits.softmax(-1)
            max_indices = probas.argmax(-1)
            num_classes = logits.size(-1)

            factor = F.one_hot(max_indices, num_classes=num_classes) - probas
            grad = (factor[:, :, None] * inputs[:, None, :])

            gradients.append(grad.cpu())
            if return_pseudo_labels:
                pseudo_labels.append(probas.argmax(-1))
            if return_embeddings:
                embeddings.append(inputs)

        # Concat all tensors and return according to the requests
        gradients = torch.cat(gradients)
        if return_embeddings:
            embeddings = torch.cat(embeddings)
            if return_pseudo_labels:
                pseudo_labels = torch.cat(pseudo_labels)
                return gradients, embeddings, pseudo_labels
            else:
                return gradients, embeddings
        elif return_pseudo_labels:
            pseudo_labels = torch.cat(pseudo_labels)
            return gradients, pseudo_labels
        else:
            return gradients