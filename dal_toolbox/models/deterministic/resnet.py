import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes, imagenethead=False):
        super().__init__()
        self.in_planes = 64
        self.block = BasicBlock
        self.num_blocks = [2, 2, 2, 2]
        self.num_classes = num_classes

        if imagenethead:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            # Init layer does not have a kernel size of 7 since cifar has a smaller size of 32x32
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * self.block.expansion, self.num_classes)

        self.dropout = nn.Dropout()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        features = out
        out = self.linear(out)
        if return_features:
            out = (out, features)
        return out
    
    def set_dropout(self, p):
        self.dropout = nn.Dropout(p=p)
    
    def forward_dropout(self, x, return_features=False):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        features = out
        out = self.linear(self.dropout(out))
        if return_features:
            out = (out, features)
        return out
    
    @torch.inference_mode()
    def get_alpha_grad_representations(self, dataloader, device):
        self.to(device)
        self.eval()

        # Create a clone of last linear layer with gradients enabled
        # TODO: Is there an easier way to do this?
        with torch.inference_mode(False), torch.autocast("cuda", enabled=False):
            linear = nn.Linear(512, 10, device='cuda')
            linear.weight = nn.Parameter(self.linear.weight.clone().requires_grad_(True))  
            linear.bias = nn.Parameter(self.linear.bias.clone().requires_grad_(True))  

        gradients, all_logits, embeddings = [], [], []
        for batch in dataloader:
            x = batch[0].to(device)
            _, emb = self(x, return_features=True)
            with torch.inference_mode(False), torch.autocast("cuda", enabled=False):
                emb = emb.clone().requires_grad_()
                logits = linear(emb)
                loss = F.cross_entropy(logits, logits.argmax(dim=1), reduction="sum")
                grad = torch.autograd.grad(loss, emb)[0]

            gradients.append(grad)
            embeddings.append(emb)
            all_logits.append(logits)

        # Concat all batches
        gradients = torch.cat(gradients)
        embeddings = torch.cat(embeddings)
        all_logits = torch.cat(all_logits)

        return gradients, embeddings, all_logits

    @torch.inference_mode()
    def get_logits(self, dataloader, device, return_features=False):
        self.to(device)
        self.eval()
        all_logits, all_features = [], []
        for batch in dataloader:
            inputs = batch[0]
            if return_features:
                logits, features = self(inputs.to(device), return_features=True)
                all_features.append(features)
            else:
                logits = self(inputs.to(device))
            all_logits.append(logits)
        logits = torch.cat(all_logits)
        if return_features:
            features = torch.cat(all_features)
            return logits, features
        return logits

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        logits = self.get_logits(dataloader=dataloader, device=device)
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
    def get_grad_representations(self, dataloader, device, return_pseudo_labels=False, return_embeddings=False):
        self.eval()
        self.to(device)

        embeddings, gradients, pseudo_labels = [], [], []
        for batch in dataloader:
            inputs = batch[0].to(device)
            logits, features = self(inputs, return_features=True)

            probas = logits.softmax(-1)
            max_indices = probas.argmax(-1)
            num_classes = logits.size(-1)

            factor = F.one_hot(max_indices, num_classes=num_classes) - probas
            grad = (factor[:, :, None] * features[:, None, :])

            gradients.append(grad.cpu())
            if return_pseudo_labels:
                pseudo_labels.append(probas.argmax(-1).cpu())
            if return_embeddings:
                embeddings.append(features.cpu())

        # Concat all tensors and return according to the requests
        gradients = torch.cat(gradients)
        if return_embeddings:
            embeddings = torch.cat(embeddings)
            if return_pseudo_labels:
                pseudo_labels = torch.cat(pseudo_labels)
                return gradients, embeddings, pseudo_labels
            else:
                return gradients, embeddings
        elif return_pseudo_labels:
            pseudo_labels = torch.cat(pseudo_labels)
            return gradients, pseudo_labels
        else:
            return gradients
        
    @torch.inference_mode()
    def get_representations_and_probas(self, dataloader):
        all_features = []
        all_probas = []
        for batch in dataloader:
            input = batch[0]
            logits, features = self(input, return_features=True)
            all_features.append(features.cpu())
            all_probas.append(logits.softmax(-1))
        features = torch.cat(all_features)
        probas = torch.cat(all_probas)
        return features, probas




# TODO (ynagel, dhuseljic) This is a lot of repeated code
class ResNet50(nn.Module):
    def __init__(self, num_classes, imagenethead=False):
        super().__init__()
        self.in_planes = 64
        self.block = Bottleneck
        self.num_blocks = [3, 4, 6, 3]
        self.num_classes = num_classes

        if imagenethead:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            # Init layer does not have a kernel size of 7 since cifar has a smaller size of 32x32
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * self.block.expansion, self.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        features = out
        out = self.linear(out)
        if return_features:
            out = (out, features)
        return out

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
        logits = self.get_logits(dataloader=dataloader, device=device)
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
        feature_dim = 512

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
