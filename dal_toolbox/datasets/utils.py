import numpy as np
from torch.utils.data import Dataset, DataLoader


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


class FeatureDataset(Dataset):
    def __init__(self, model, dataset, device):
        dataloader = DataLoader(dataset, batch_size=512, num_workers=4)
        features, labels = model.get_representations(dataloader, device, return_labels=True)
        self.features = features.detach()
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def sample_balanced_subset(targets, num_classes, num_samples):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    # Get samples per class
    assert num_samples % num_classes == 0, "lb_num_labels must be divideable by num_classes in balanced setting"
    lb_samples_per_class = [int(num_samples / num_classes)] * num_classes

    val_pool = []
    for c in range(num_classes):
        idx = np.array([i for i in range(len(targets)) if targets[i] == c])
        np.random.shuffle(idx)
        val_pool.extend(idx[:lb_samples_per_class[c]])
    return [int(i) for i in val_pool]
