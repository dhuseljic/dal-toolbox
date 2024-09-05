import torch
import torch.nn as nn
from dal_toolbox.models.utils.mcdropout import MCDropoutModule, ConsistentMCDropout



    

class SimpleMCNet(MCDropoutModule):
    def __init__(self,
                 num_classes: int,
                 dropout_rate: int = .2,
                 feature_dim: int = 128,
                 ):
        super().__init__(n_passes=50)

        self.first = nn.Linear(2, feature_dim)
        self.first_dropout = ConsistentMCDropout(dropout_rate)
        self.hidden = nn.Linear(feature_dim, feature_dim)
        self.hidden_dropout = ConsistentMCDropout(dropout_rate)
        self.last = nn.Linear(feature_dim, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.first(x))
        x = self.first_dropout(x)
        x = self.act(self.hidden(x))
        x = self.hidden_dropout(x)
        out = self.last(x)
        return out

    def get_logits(self, dataloader, device):
        device='cpu'
        self.to(device)
        mc_logits_list = []
        for batch in dataloader:
            mc_logits_list.append(self.mc_forward(batch[0].to(device)).cpu())
        return torch.cat(mc_logits_list)
    
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
        self.eval()
        self.to(device)

        embedding = []
        for batch in dataloader:
            inputs = batch[0]
            embedding_batch = torch.empty([len(inputs), self.in_dimension * self.num_classes])
            logits = self(inputs.to(device)).cpu()
            features = inputs.cpu()

            probas = logits.softmax(-1)
            max_indices = probas.argmax(-1)

            for n in range(len(inputs)):
                for c in range(self.num_classes):
                    if c == max_indices[n]:
                        embedding_batch[n, self.in_dimension * c: self.in_dimension * (c + 1)] = \
                            features[n] * (1 - probas[n, c])
                    else:
                        embedding_batch[n, self.in_dimension * c: self.in_dimension * (c + 1)] = \
                            features[n] * (-1 * probas[n, c])
            embedding.append(embedding_batch)
        # Concat all embeddings
        embedding = torch.cat(embedding)
        return embedding