import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from enums import DataSplit


class RandomDataset(Dataset):
    def __init__(self, length=32, data_split: DataSplit = DataSplit.TRAIN):
        super().__init__()
        self.length = length
        self.data_split = data_split
        if self.data_split == DataSplit.TRAIN:
            self.size = 1000
        else:
            self.size = 150

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random vector as input, random label (binary classification)
        x = torch.randn(self.length)
        y = torch.randint(0, 2, ())
        return x, y


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            RandomDataset(data_split=DataSplit.TRAIN),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            RandomDataset(data_split=DataSplit.VAL),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            RandomDataset(data_split=DataSplit.VAL),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
