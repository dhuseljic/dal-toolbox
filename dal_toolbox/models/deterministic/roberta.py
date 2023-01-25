import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm

class RoBertaSequenceClassifier(nn.Module):
    def __init__(self, checkpoint, num_classes):
        super(RoBertaSequenceClassifier, self).__init__()

        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint,
            num_labels=self.num_classes)
            
    def forward(self, input_ids, attention_mask, return_cls=False):
        outputs = self.roberta(input_ids, attention_mask, labels=None, output_hidden_states=True)
        logits = outputs['logits']

        last_hidden_state = outputs['hidden_states'][-1]
        cls_state = last_hidden_state[:,0,:]
        if return_cls:
            output = (logits, cls_state)
        else:
            output = logits
        return output

    @torch.no_grad()
    def forward_logits(self, dataloader, device):
        self.to(device)
        all_logits = []
        for samples, _ in dataloader:
            logits = self(samples.to(device))
            all_logits.append(logits)
        return torch.cat(all_logits)

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        self.to(device)
        self.eval()
        all_logits = []
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = self(input_ids, attention_mask)
            all_logits.append(logits.to("cpu"))
        logits = torch.cat(all_logits)
        probas = logits.softmax(-1)
        return probas  
    
    @torch.inference_mode()
    def get_representation(self, dataloader, device):
        self.to(device)
        self.eval()
        all_features = []
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _, cls_state = self(input_ids, attention_mask, return_cls=True)
            all_features.append(cls_state.to("cpu"))
        features = torch.cat(all_features)
        return features

    @torch.inference_mode()
    def get_grad_embedding(self, dataloader, device):
        self.eval()
        self.to(device)
        feature_dim = 768

        embedding = []
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)           
            embedding_batch = torch.empty([len(input_ids), feature_dim * self.num_classes])
            logits, cls_state = self(input_ids, attention_mask, return_cls=True)
            logits = logits.cpu()
            features = cls_state.cpu()

            probas = logits.softmax(-1)
            max_indices = probas.argmax(-1)

            # TODO: optimize code
            # for each sample in a batch and for each class, compute the gradient wrt to weights
            for n in range(len(input_ids)):
                for c in range(self.num_classes):
                    if c == max_indices[n]:
                        embedding_batch[n, feature_dim * c: feature_dim * (c+1)] = features[n] * (1 - probas[n, c])
                    else:
                        embedding_batch[n, feature_dim * c: feature_dim * (c+1)] = features[n] * (-1 * probas[n, c])
            embedding.append(embedding_batch)
        # Concat all embeddings
        embedding = torch.cat(embedding)
        return embedding