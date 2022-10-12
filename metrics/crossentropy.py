import torch
import torch.nn as nn
import math

class EnsembleCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: list, labels: list):
        """
        logits.shape = (EnsembleMembers, Samples, Classes)
        labels.shape = (Samples)
        """
        # Remember ensemble_size for nll calculation
        ensemble_size, n_samples, n_classes = logits.shape

        # Reshape to fit CrossEntropy 
        # logits.shape = (EnsembleMembers*Samples, Classes)
        # labels.shape = (EnsembleMembers*Sampels)
        labels = torch.broadcast_to(labels.unsqueeze(0), logits.shape[:-1])
        labels = labels.reshape(ensemble_size*n_samples)
        logits = logits.reshape(ensemble_size*n_samples, n_classes)
        
        # Non Reduction Cross Entropy
        loss = torch.nn.CrossEntropyLoss(reduction='none')
        ce = loss(logits, labels).reshape(-1, 1)

        # Reduce LogSumExp + log of Ensemble Size
        nll = -torch.logsumexp(-ce, dim=1) + math.log(ensemble_size)

        #Return Average
        return torch.mean(nll)


#TODO: Change according to robustness metrics

class GibsCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: list, labels: list):
        """
        logits.shape = (EnsembleMembers, Samples, Classes)
        labels.shape = (Samples)
        """
        # Remember ensemble_size for nll calculation
        ensemble_size, n_samples, n_classes = logits.shape

        # Reshape to fit CrossEntropy 
        # logits.shape = (EnsembleMembers*Samples, Classes)
        # labels.shape = (EnsembleMembers*Sampels)
        labels = torch.broadcast_to(labels.unsqueeze(0), logits.shape[:-1])
        labels = labels.reshape(ensemble_size*n_samples)
        logits = logits.reshape(ensemble_size*n_samples, n_classes)
        
        # Non Reduction Cross Entropy
        loss = torch.nn.CrossEntropyLoss(reduction='none')
        ce = loss(logits, labels).reshape(-1, 1)

        # Reduce LogSumExp + log of Ensemble Size
        nll = -torch.logsumexp(-ce, dim=1) + math.log(ensemble_size)

        #Return Average
        return torch.mean(nll)