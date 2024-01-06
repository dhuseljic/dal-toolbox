import torch
import torch.nn as nn


class ViTsWrapper(nn.Module):
    def __init__(self, encoder, encoder_output_dim, model_params, num_classes):
        super(ViTsWrapper, self).__init__()

        self.encoder = encoder
        self.encoder_output_dim = encoder_output_dim
        self.encoder.load_state_dict(model_params)
        self.num_classes = num_classes
        self.linear = nn.Linear(self.encoder_output_dim, self.num_classes)

    def forward(self, x, return_features=False):
        out = self.encoder(x)
        features = out
        out = self.linear(out)
        if return_features:
            out = (out, features)
        return out

    @torch.no_grad()
    def forward_logits(self, dataloader, device):
        self.to(device)
        all_logits = []
        for samples, _ in dataloader:
            logits = self(samples.to(device))
            all_logits.append(logits)
        return torch.cat(all_logits)

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
        self.to(device)
        self.eval()
        all_logits = []
        for samples, _ in dataloader:
            logits = self(samples.to(device))
            all_logits.append(logits)
        logits = torch.cat(all_logits)
        probas = logits.softmax(-1)
        return probas

    @torch.inference_mode()
    def get_representations(self, dataloader, device, return_labels=False):
        self.to(device)
        self.eval()
        all_features = []
        all_labels = []
        for batch in dataloader:
            inputs = batch[0]
            labels = batch[1]
            _, features = self(inputs.to(device), return_features=True)
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
        feature_dim = self.encoder_output_dim

        embedding = []
        for batch in dataloader:
            inputs = batch[0]
            embedding_batch = torch.empty([len(inputs), feature_dim * self.num_classes])
            logits, features = self(inputs.to(device), return_features=True)
            logits = logits.cpu()
            features = features.cpu()

            probas = logits.softmax(-1)
            max_indices = probas.argmax(-1)

            # TODO: optimize code
            # for each sample in a batch and for each class, compute the gradient wrt to weights
            for n in range(len(inputs)):
                for c in range(self.num_classes):
                    if c == max_indices[n]:
                        embedding_batch[n, feature_dim * c: feature_dim * (c + 1)] = features[n] * (1 - probas[n, c])
                    else:
                        embedding_batch[n, feature_dim * c: feature_dim * (c + 1)] = features[n] * (-1 * probas[n, c])
            embedding.append(embedding_batch)
        # Concat all embeddings
        embedding = torch.cat(embedding)
        return embedding
