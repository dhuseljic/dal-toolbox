import torch
import math

# TODO: Transform into class

def nll(logits, labels):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    ensemble_size = logits.shape[0]
    # Broadcast Labels to fit the ensemble logits
    labels = torch.broadcast_to(labels, logits.shape[:-1])

    # Sparse Softmax Cross Entropy with logits which equals calculate
    # Cross Entropy Loss but for each logit label pair individually
    ce = loss(logits, labels)

    # Reduce LogSumExp + log of Ensemble Size
    nll = -torch.logsumexp(-ce, dim=0) + math.log(ensemble_size)

    #Average
    nll = torch.mean(nll)

    return nll.item()