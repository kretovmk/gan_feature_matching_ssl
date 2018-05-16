
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
        

def get_data_loaders(name='cifar10', batch_size=32, use_cuda=False, download=True, num_workers=0, drop_last=True):
    root = './data'
    trans = transforms.ToTensor()
    if name == 'mnist':
        train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
        test_set = dset.MNIST(root=root, train=False, transform=trans, download=download)
    elif name == 'cifar10':
        train_set = dset.CIFAR10(root=root, train=True, transform=trans, download=download)
        test_set = dset.CIFAR10(root=root, train=False, transform=trans, download=download)
    kwargs = {'num_workers': num_workers, 'pin_memory': use_cuda, 'drop_last': drop_last}
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader   