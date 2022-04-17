from torch.utils.data import DataLoader
from torchvision.utils import make_grid

def plot_grids(train_ds, test_ds_id, test_ds_ood):
    import pylab as plt
    get_batch = lambda dataset: next(iter(DataLoader(dataset, batch_size=32)))

    fig = plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('Training Samples')
    grid = make_grid(get_batch(train_ds)[0])
    plt.imshow(grid.permute(1, 2, 0))
    plt.subplot(132)
    plt.title('Test Samples ID')
    grid = make_grid(get_batch(test_ds_id)[0])
    plt.imshow(grid.permute(1, 2, 0))
    plt.subplot(133)
    plt.title('Test Samples OOD')
    grid = make_grid(get_batch(test_ds_ood)[0])
    plt.imshow(grid.permute(1, 2, 0))
    return fig