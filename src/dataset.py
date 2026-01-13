import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import lightning as L

from src.utils import SEED


L.seed_everything(SEED)


class GasDataModule(L.LightningDataModule):
    def __init__(
            self,
            train_data: tuple | None = None,
            val_data: tuple | None = None,
            test_data: tuple | None = None,
            batch_size: int = 64,
        ):
        super().__init__()

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size
        else:
            self.batch_size_per_device = self.batch_size

        if stage == "fit" or stage is None:
            if self.train_data is not None:
                self.train_dataset = list(zip(*self.train_data))
            if self.val_data is not None:
                self.val_dataset = list(zip(*self.val_data))
            
        if stage == "test" or stage is None:
            if self.test_data is not None:
                self.test_dataset = list(zip(*self.test_data))

        elif stage == 'predict':
            self.pred_dataset = 1
            pass

    def _collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.tensor(np.stack(x), dtype=torch.float32)

        # # Index
        # y = torch.tensor(y, dtype=torch.int64)

        # One-Hot Encoding
        y = torch.tensor(np.stack(y), dtype=torch.float32)
        return x, y
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_per_device, shuffle=True, pin_memory=True, collate_fn=self._collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size_per_device, shuffle=False, pin_memory=True, collate_fn=self._collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_per_device, shuffle=False, pin_memory=True, collate_fn=self._collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size_per_device, shuffle=False, collate_fn=self._collate_fn)
    