import os
import glob
import argparse
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

from src.model import create_model
from src.dataset import GasDataModule
from src.utils import SEED


L.seed_everything(SEED)


class GasClsModel(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        input_length: int,
        num_classes: int = 3,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(
            model=model_name,
            input_length=input_length,
            num_classes=num_classes,
        )

        # One-Hot Encoding loss_fn
        self.criterion = nn.BCEWithLogitsLoss()
        
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

        # # Index
        # loss = F.cross_entropy(logits, y)
        # acc = (logits.argmax(dim=1) == y).float().mean()

        # One-Hot Encoding
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

        # # Index
        # loss = F.cross_entropy(logits, y)
        # acc = (logits.argmax(dim=1) == y).float().mean()

        # One-Hot Encoding
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(y, dim=1)
        acc = (preds == targets).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # # Index
        # preds = torch.argmax(logits, dim=1)
        # loss = F.cross_entropy(logits, y)
        # acc = (logits.argmax(dim=1) == y).float().mean()

        # One-Hot Encoding
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
        self.log("test/acc", self.test_acc, prog_bar=True)
        self.log("test/precision", self.test_precision)
        self.log("test/recall", self.test_recall)
        self.log("test/f1", self.test_f1)

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).numpy()
        targets = torch.cat(self.test_targets).numpy()
        cm = confusion_matrix(targets, preds)

        class_names = ["acetone", "benzene", "toluene"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix")

        save_path = os.path.join("./result", f"cm_{args.model_name}_{datetime.now().strftime("%m%d_%H%M%S")}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        # reset (안 하면 여러 test run 시 누적됨)
        self.test_preds.clear()
        self.test_targets.clear()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def build_samples(df: dict, gas_to_label: dict):
    samples = []
    labels = []

    for gas, label in gas_to_label.items():
        merge = df[gas]["merge"].to_numpy()
        time = merge[:, 0]
        signal = merge[:, 1:]

        for i in range(signal.shape[1]):
            samples.append(signal[:, i].astype(np.float32))
            labels.append(label)

    x = np.stack(samples)
    num_classes = len(gas_to_label)
    y_index = np.array(labels)
    y_onehot = np.eye(num_classes, dtype=np.float32)[labels]

    print("x shape", x.shape)
    print("y_index shape", y_index.shape)
    print("y_onehot shape", y_onehot.shape)
    return x, y_index, y_onehot

def main(args):
    os.makedirs(args.save, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath= f'{args.save}',
        filename= f'{args.model_name}-'+'{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=15
    )
    wandb_logger = WandbLogger(
        project="Gas",
        name=f"{args.model_name}_{datetime.now().strftime("%m%d_%H%M%S")}"
    )
    
    gas_to_label = {
        "acetone": 0,
        "benzene": 1,
        "toluene": 2,
    }
    
    df = {}
    for path in glob.glob("data/pkl/*.pkl"):
        filename = os.path.splitext(os.path.basename(path))[0]
        gas, data_type = filename.split("_", 1)

        if gas not in df:
            df[gas] = {}

        with open(path, "rb") as f:
            obj = pickle.load(f)

        df[gas][data_type] = obj
    X, y_index, y_onehot = build_samples(df, gas_to_label)

    # skf = StratifiedKFold(
    #     n_splits=5,
    #     shuffle=True,
    #     random_state=SEED
    # )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot,
        test_size=0.2,
        random_state=SEED,
        stratify=y_onehot,
    )

    train_data = (X_train, y_train)
    test_data = (X_test, y_test)

    trainer = L.Trainer(
        accelerator=args.device,
        devices=1,
        max_epochs=args.epoch,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
    )

    if args.mode == 'train':
        model = GasClsModel(args.model_name, input_length=7300)
        trainer.fit(model, GasDataModule(train_data, args.batch))
    elif args.mode == 'test':
        model = GasClsModel.load_from_checkpoint(
            args.ckpt,
            model=create_model(
                model=args.model_name,
                input_length=7300,
                num_classes=3,
            ),
            num_classes=3,
        )

        trainer.test(model, GasDataModule(test_data, args.batch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='./data/pkl')
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')
    parser.add_argument('-c', '--ckpt_path', dest='ckpt', type=str, default='./checkpoint/')
    parser.add_argument('-mn', '--model_name', type=str, default='cnn1d')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=300)
    parser.add_argument('-dv', '--device', type=str, default='gpu')
    parser.add_argument('-g', '--gpus', type=str, nargs='+', default='0')
    parser.add_argument('-m', '--mode', type=str, default='train')
    args = parser.parse_args()
    
    main(args)
