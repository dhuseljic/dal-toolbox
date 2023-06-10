import numpy as np


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


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
