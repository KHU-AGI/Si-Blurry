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
from .OnlineIterDataset import OnlineIterDataset
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
    # "grayCUB200",
    # "grayFlowers102",
    # "graySVHN",
    # "grayTinyImageNet",
    # "grayCIFAR10",
    # "grayCIFAR100",
    "OnlineIterDataset",
]