
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def plot_grid(samples, num=2, img_size=(28, 28)):
    fig = plt.figure(figsize=(num, num))
    gs = gridspec.GridSpec(num, num)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(*img_size), cmap='Greys_r')


def get_data_loaders(name='cifar10', batch_size=32, use_cuda=False, download=True, num_workers=0, 
                     transform=transforms.ToTensor(), drop_last=True, train_size=-1, num_classes=10):
    root = './data'
    if name == 'mnist':
        train_set = dset.MNIST(root=root, train=True, transform=transform, download=download)
        test_set = dset.MNIST(root=root, train=False, transform=transform, download=download)
    elif name == 'cifar10':
        train_set = dset.CIFAR10(root=root, train=True, transform=transform, download=download)
        test_set = dset.CIFAR10(root=root, train=False, transform=transform, download=download)
    if train_size > 0:
        tx = []
        ty = []
        ix = np.array(train_set.train_labels)
        ones = np.ones(train_size).astype(int)
        for i in range(num_classes):
            x = train_set.train_data[ix == i][:train_size]
            tx.append(x)
            y = list(ones * i)
            ty += y
        train_set.train_data = np.concatenate(tx, axis=0)
        train_set.train_labels = ty         
    kwargs = {'num_workers': num_workers, 'pin_memory': use_cuda, 'drop_last': drop_last}
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader   


def log_sum_exp(tensor, keepdim=True):
    r"""
    Numerically stable implementation for the `LogSumExp` operation. The
    summing is done along the last dimension.
    Args:
        tensor (torch.Tensor)
        keepdim (Boolean): Whether to retain the last dimension on summing.
    """
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    return max_val + (tensor - max_val).exp().sum(dim=-1, keepdim=keepdim).log()