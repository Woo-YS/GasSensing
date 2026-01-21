import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L

class GasDataModule(L.LightningDataModule):
    def __init__(
            self,
            train_data = None, # list(zip) 또는 tuple(X, y) 모두 허용
            val_data = None,
            test_data = None,
            batch_size: int = 64,
            task: str = 'reg'  # ⚡ 'reg' 또는 'cls' (필수)
        ):
        super().__init__()
        self.train_data_raw = train_data
        self.val_data_raw = val_data
        self.test_data_raw = test_data
        self.batch_size = batch_size
        self.task = task

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _prepare_data(self, data):
        """
        입력이 튜플 (X, y)이면 -> list(zip(X, y))로 변환 (분류용)
        입력이 이미 list 이면 -> 그대로 사용 (회귀용)
        """
        if data is None:
            return None
        
        # Classification 스타일 (X_array, y_array) 튜플인 경우
        if isinstance(data, tuple):
            return list(zip(*data))
        
        # Regression 스타일 (이미 리스트)인 경우
        return data

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._prepare_data(self.train_data_raw)
            self.val_dataset = self._prepare_data(self.val_data_raw)
            
        if stage == "test" or stage is None:
            self.test_dataset = self._prepare_data(self.test_data_raw)

    def _collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.tensor(np.stack(x), dtype=torch.float32)
        
        # ⚡ Task에 따라 y의 모양(Shape) 결정
        if self.task == 'reg':
            # 회귀: [Batch, 1] (MSE Loss용)
            y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
        else:
            # 분류: [Batch, Classes] (One-Hot) 또는 [Batch] (Index)
            # 팀원 코드의 One-Hot Encoding 유지를 위해 stack 사용
            y = torch.tensor(np.stack(y), dtype=torch.float32)
            
        return x, y
    
    def train_dataloader(self):
        if self.train_dataset:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn)
    
    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn)
    
    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn)