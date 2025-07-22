import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import pandas as pd
from torch.utils.data import Dataset

class MNIST_CSV(Dataset):
    def __init__(self, csv_path, train=True, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.train = train

        if self.train:
            self.labels = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.long)
            self.images = torch.tensor(self.data.iloc[:, 1:].values, dtype=torch.float32)
        else:
            self.labels = None
            self.images = torch.tensor(self.data.values, dtype=torch.float32)

        self.images = self.images.view(-1, 1, 28, 28) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        if self.train:
            return img, self.labels[idx]
        else:
            return img, torch.tensor(-1)