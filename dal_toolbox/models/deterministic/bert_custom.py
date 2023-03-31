import torch
import torch.nn as nn
import transformers
from transformers import AutoModel
from tqdm.auto import tqdm
from ...metrics import generalization 
from ...utils import MetricLogger, SmoothedValue

class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, X):
        hidden_logits = self.dropout(self.relu(self.layer1(X)))
        out_logits = self.layer2(hidden_logits)
        return out_logits


class BertClassifier(nn.Module):
    def __init__(self, model_name, num_classes, mode='non-static'):
        super(BertClassifier, self).__init__()

        self.mode = mode
        self.model_name = model_name
        self.num_classes = num_classes

        # no pre-training
        if self.mode == 'random':
            self.encoder = AutoModel.from_config(config=transformers.DistilBertConfig())
        
        # pre-training with freezed weights
        elif self.mode == 'static':
            self.encoder = AutoModel.from_pretrained(self.model_name)
            for param in self.encoder.parameters():
                param.required_grad = False
        
        # pre-training and training of all layers
        elif self.mode == 'non-static':
            self.encoder = AutoModel.from_pretrained(self.model_name)
        
        else:
            raise NotImplementedError("f{self.mode} is not available")
      
        self.head = ClassificationHead(768, self.num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(
            input_ids,
            attention_mask,
            return_dict=False
        )

        last_hidden_state = output[0]
        out_pooled = last_hidden_state[:, 0]
        out_logits = self.head(out_pooled)

        return out_logits
    
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
                    tokenizer,
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
        targets = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)

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
def eval_one_epoch(model, dataloader, epoch, criterion, tokenizer, device, print_freq=25):
    model.eval()
    model.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    header = "Testing:"
    for batch in metric_logger.log_every(dataloader, print_freq, header):

        targets = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)

        batch_acc, = generalization.accuracy(logits, targets)
        metric_logger.update(loss=loss.item())
        metric_logger.meters["batch_acc"].update(batch_acc.item(), n=batch_size)
    test_stats = {f"test_{name}_epoch": meter.global_avg for name, meter, in metric_logger.meters.items()}
    print(f"Epoch [{epoch}]: Test Loss: {test_stats['test_loss_epoch']:.4f}, Test Accuracy: {test_stats['test_batch_acc_epoch']:.4f}")
    print("--"*40)
    return test_stats
