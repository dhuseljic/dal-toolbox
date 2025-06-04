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

    def forward(self, x, return_features=False, apply_dropout=False):
        if apply_dropout:
            x = self.dropout(x)
        out = self.linear(x)
        if return_features:
            out = (out, x)
        return out

    # No context manager applied as alfamix requires grad calculations
    def get_alfa_grad_representations(self, dataloader, device):
        self.to(device)

        embeddings, gradients, pseudo_labels = [], [], []
        for batch in dataloader:
            embedding = batch[0].to(device).requires_grad_()
            logits = self(embedding)
            preds = logits.softmax(-1).argmax(-1)
            loss = F.cross_entropy(logits, preds, reduction="sum")
            grads = torch.autograd.grad(loss, embedding)[0]

            embeddings.append(embedding.cpu().detach())
            gradients.append(grads.cpu())
            pseudo_labels.append(preds.cpu())

        # Concat all tensors and return
        gradients = torch.cat(gradients)
        embeddings = torch.cat(embeddings)
        pseudo_labels = torch.cat(pseudo_labels)
        
        return gradients, embeddings, pseudo_labels

    @torch.inference_mode()
    def get_logits(self, dataloader, device, apply_dropout=False):
        self.to(device)
        self.train(apply_dropout)
        all_logits = []
        for batch in dataloader:
            inputs = batch[0].to(device)
            logits = self(inputs, apply_dropout=apply_dropout)
            all_logits.append(logits)
        logits = torch.cat(all_logits)
        return logits

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
    def get_representations_and_logits(self, dataloader, device):
        self.to(device)

        all_features, all_logits = [], []
        for batch in dataloader:
            features = batch[0].to(device)
            logits = self(features)
            all_features.append(features.cpu())
            all_logits.append(logits.cpu())
            
        features = torch.cat(all_features)
        logits = torch.cat(all_logits)
        return features, logits

    @torch.inference_mode()
    def get_grad_representations(self, dataloader, device):
        self.eval()
        self.to(device)

        gradients = []
        for batch in dataloader:
            inputs = batch[0].to(device)
            logits = self(inputs)

            probas = logits.softmax(-1)
            max_indices = probas.argmax(-1)
            num_classes = logits.size(-1)

            factor = F.one_hot(max_indices, num_classes=num_classes) - probas
            grad = (factor[:, :, None] * inputs[:, None, :]).flatten(-2)

            gradients.append(grad.cpu())

        # Concat all tensors and return according to the requests
        gradients = torch.cat(gradients)
        return gradients