import torch
import torch.nn.functional as F
import lightning as L


class EnsembleModule(L.LightningModule):
    def __init__(self, members: list):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.automatic_optimization = False

    def training_step(self, batch):
        inputs, targets = batch
        for i_member, (member, optimizer) in enumerate(zip(self.members, self.optimizers())):
            logits = member(inputs)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            self.log(f'loss_member{i_member}', loss, prog_bar=True)

    def forward(self, x):
        logits_list = []
        for member in self.members:
            logits_list.append(member(x))
        return torch.stack(logits_list, dim=1)
