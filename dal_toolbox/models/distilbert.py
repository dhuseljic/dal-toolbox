import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm
from ..metrics import generalization 
from ..utils import MetricLogger, SmoothedValue
#!TODO: muss noch gemacht werden

class DistilbertSequenceClassifier(nn.Module):
    def __init__(self, checkpoint, num_classes):
        super(DistilbertSequenceClassifier, self).__init__()

        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint, 
            num_labels=self.num_classes)
            
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None, 
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.roberta(
            input_ids, 
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            labels,
            output_attentions,
            output_hidden_states)

        logits = outputs['logits']

        return logits

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

def train_one_epoch(model,
                    dataloader,
                    epoch,
                    optimizer,
                    scheduler,
                    criterion,
                    device,
                    print_freq=25):
    model.train()
    model.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter(
        "lr", 
        SmoothedValue(window_size=1, 
        fmt="{value:.8f}")
    )
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    for batch in metric_logger.log_every(dataloader, print_freq, header):
        batch = batch.to(device)
        targets = batch['labels']
       
        logits = model(**batch)

        loss = criterion(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        batch_size = targets.size(0)
        batch_acc, = generalization.accuracy(logits, targets, topk=(1,))
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["batch_acc"].update(batch_acc.item(), n=batch_size)

    # save global (epoch) stats: take average of all the saved batch 
    train_stats = {f"train_{name}_epoch": meter.global_avg for name, meter, in metric_logger.meters.items()}
    print(f"Epoch [{epoch}]: Train Loss: {train_stats['train_loss_epoch']:.4f}, Train Accuracy: {train_stats['train_batch_acc_epoch']:.4f}")
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
       
        logits = model(**batch)

        loss = criterion(logits, targets)

        batch_size = targets.size(0)

        batch_acc, = generalization.accuracy(logits, targets)
        metric_logger.update(loss=loss.item())
        metric_logger.meters["batch_acc"].update(batch_acc.item(), n=batch_size)
    test_stats = {f"test_{name}_epoch": meter.global_avg for name, meter, in metric_logger.meters.items()}
    print(f"Epoch [{epoch}]: Test Loss: {test_stats['test_loss_epoch']:.4f}, Test Accuracy: {test_stats['test_batch_acc_epoch']:.4f}")
    print("--"*40)
    return test_stats