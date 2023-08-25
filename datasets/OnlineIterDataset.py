# Multi datasets for continual learning
# All datasets needs to be in the same format.
# have targets and classes within the dataset.

from typing import Callable, Optional, Iterable
from torch.utils.data import Dataset

class OnlineIterDataset(Dataset):
    def __init__(self,
                 dataset   : Dataset,
                 iteration : int,
                 ) -> None:
        super().__init__()
        self.dataset = dataset
        self.iteration = int(iteration)
        self.classes = dataset.classes
        self.targets = dataset.targets

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        return image, label, index

    def __len__(self):
        return len(self.dataset)