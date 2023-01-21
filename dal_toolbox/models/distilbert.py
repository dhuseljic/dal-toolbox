import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm
from ..metrics import generalization 
from ..utils import MetricLogger, SmoothedValue

class DistilbertSequenceClassifier(nn.Module):
    def __init__(self, checkpoint, num_classes):
        super(DistilbertSequenceClassifier, self).__init__()

        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.distilbert = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint,
            num_labels=self.num_classes)
            
    def forward(self, input_ids, attention_mask, return_cls=False):
        outputs = self.distilbert(input_ids, attention_mask, labels=None, output_hidden_states=True)
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

def train_one_epoch(model, dataloader, epoch, optimizer, scheduler, criterion, device, print_freq=25):
    model.train()
    model.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.8f}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    for batch in metric_logger.log_every(dataloader, print_freq, header):
        batch = batch.to(device)
        targets = batch['labels']
       
        logits = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        batch_size = targets.size(0)
        batch_acc, = generalization.accuracy(logits, targets, topk=(1,))
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["batch_acc"].update(batch_acc.item(), n=batch_size)

    # save global (epoch) stats: take average of all the saved batch 
    train_stats = {f"train_{name}_epoch": meter.global_avg for name, meter, in metric_logger.meters.items()}
    print(f"Epoch [{epoch}]: Train Loss: {train_stats['train_loss_epoch']:.4f}, \
        Train Accuracy: {train_stats['train_batch_acc_epoch']:.4f}")
    print("--"*40)
    return train_stats

@torch.no_grad()
def eval_one_epoch(model, dataloader, epoch, criterion, device, print_freq=25):
    model.eval()
    model.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    header = "Testing:"
    for batch in metric_logger.log_every(dataloader, print_freq, header):
        batch = batch.to(device)
        targets = batch['labels']
       
        logits = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        batch_acc, = generalization.accuracy(logits, targets)
        batch_f1 = generalization.f1_macro(logits, targets, model.num_classes, device)
        batch_acc_balanced = generalization.balanced_acc(logits, targets, device)

        metric_logger.update(loss=loss.item())
        metric_logger.meters['batch_acc'].update(batch_acc.item(), n=batch_size)
        metric_logger.meters['batch_f1'].update(batch_f1.item(), n=batch_size)
        metric_logger.meters['batch_acc_balanced'].update(batch_acc_balanced.item(), n=batch_size)
        
    test_stats = {f"test_{name}_epoch": meter.global_avg for name, meter, in metric_logger.meters.items()}
    print(f"Epoch [{epoch}]: Test Loss: {test_stats['test_loss_epoch']:.4f}, \
        Test Accuracy: {test_stats['test_batch_acc_epoch']:.4f}")
    print("--"*40)
    return test_stats
