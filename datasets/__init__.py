from .CUB200 import CUB200
from .multiDatasets import multiDatasets
from .Flowers102 import Flowers102
from .NotMNIST import NotMNIST
from .SVHN import SVHN
from .TinyImageNet import TinyImageNet
# from .grayCUB200 import grayCUB200
# from .grayFlowers102 import grayFlowers102
# from .graySVHN import graySVHN
# from .grayTinyImageNet import grayTinyImageNet
# from .grayCIFAR10 import grayCIFAR10
# from .grayCIFAR100 import grayCIFAR100
from .MNIST import MNIST
from .FashionMNIST import FashionMNIST
from .Imagenet_R import Imagenet_R
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet

__all__ = [
    "CUB200",
    "multiDatasets",
    "Flowers102",
    "NotMNIST",
    "SVHN",
    "TinyImageNet",
    "CIFAR10",
    "CIFAR100",
    "MNIST",
    "FashionMNIST",
    "ImageNet",
    "Imagenet_R",
]

# dictionary of tuple of dataset, mean, std
datasets = {
    "cifar10": (CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": (CIFAR100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "svhn": (SVHN, (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "fashionmnist": (FashionMNIST, (0.2860,), (0.3530,)),
    "mnist": (MNIST, (0.1307,), (0.3081,)),
    "tinyimagenet": (TinyImageNet, (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    "notmnist": (NotMNIST, (0.1307,), (0.3081,)),
    "cub200": (CUB200, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "imagenet": (ImageNet, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "imagenet-r": (Imagenet_R, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}

def get_dataset(name):
    return datasets[name]