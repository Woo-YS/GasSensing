import os
import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
)
from src.model import create_model

class GasClsModel(L.LightningModule):
    def __init__(self, model_name, input_length, num_classes=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model=model_name, input_length=input_length, output_dim=num_classes)
        self.criterion = nn.BCEWithLogitsLoss()

        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(y, dim=1)
        acc = (preds == targets).float().mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(y, dim=1)
        
        self.val_acc(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(y, dim=1)

        self.test_acc(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)
        
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1)

    def on_test_epoch_end(self):
        if not self.test_preds: return
        preds = torch.cat(self.test_preds).numpy()
        targets = torch.cat(self.test_targets).numpy()
        cm = confusion_matrix(targets, preds)
        
        class_names = ["acetone", "benzene", "toluene"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix ({self.hparams.model_name})")
        
        os.makedirs("./result", exist_ok=True)
        save_path = os.path.join("./result", f"cm_{self.hparams.model_name}_{datetime.now().strftime('%m%d_%H%M%S')}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        self.test_preds.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)