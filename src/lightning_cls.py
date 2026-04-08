# import os
# import torch
# import torch.nn as nn
# import lightning as L
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from datetime import datetime
# from sklearn.metrics import confusion_matrix
# from torchmetrics.classification import (
#     MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
# )
# from src.model import create_model

# class GasClsModel(L.LightningModule):
#     def __init__(self, model_name, input_length, num_classes=3, lr=1e-3):
#         super().__init__()
#         self.save_hyperparameters()
#         self.model = create_model(model=model_name, input_length=input_length, output_dim=num_classes)
#         self.criterion = nn.BCEWithLogitsLoss()

#         self.val_acc = MulticlassAccuracy(num_classes=num_classes)
#         self.val_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
#         self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
#         self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        
#         self.test_acc = MulticlassAccuracy(num_classes=num_classes)
#         self.test_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
#         self.test_recall = MulticlassRecall(num_classes=num_classes, average="macro")
#         self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

#         self.test_preds = []
#         self.test_targets = []

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.criterion(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         targets = torch.argmax(y, dim=1)
#         acc = (preds == targets).float().mean()
#         self.log("train_loss", loss, on_epoch=True, prog_bar=True)
#         self.log("train_acc", acc, on_epoch=True, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.criterion(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         targets = torch.argmax(y, dim=1)
        
#         self.val_acc(preds, targets)
#         self.val_precision(preds, targets)
#         self.val_recall(preds, targets)
#         self.val_f1(preds, targets)
        
#         self.log("val_loss", loss, prog_bar=True, sync_dist=True)
#         self.log("val_acc", self.val_acc, prog_bar=True, sync_dist=True)

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.criterion(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         targets = torch.argmax(y, dim=1)

#         self.test_acc(preds, targets)
#         self.test_precision(preds, targets)
#         self.test_recall(preds, targets)
#         self.test_f1(preds, targets)
        
#         self.test_preds.append(preds.detach().cpu())
#         self.test_targets.append(targets.detach().cpu())

#         self.log("test_loss", loss)
#         self.log("test_acc", self.test_acc)
#         self.log("test_f1", self.test_f1)

#     def on_test_epoch_end(self):
#         if not self.test_preds: return
#         preds = torch.cat(self.test_preds).numpy()
#         targets = torch.cat(self.test_targets).numpy()
#         cm = confusion_matrix(targets, preds)
        
#         class_names = ["acetone", "benzene", "toluene"]
#         plt.figure(figsize=(6, 5))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
#         plt.title(f"Confusion Matrix ({self.hparams.model_name})")
        
#         os.makedirs("./result", exist_ok=True)
#         save_path = os.path.join("./result", f"cm_{self.hparams.model_name}_{datetime.now().strftime('%m%d_%H%M%S')}.png")
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=300)
#         plt.close()
        
#         self.test_preds.clear()
#         self.test_targets.clear()

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



import os
import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd  # 데이터 분석용
from datetime import datetime
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
)
from src.model import create_model # models.py에서 create_model 불러오기

class GasClsModel(L.LightningModule):
    def __init__(self, model_name, input_length, num_classes=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # 모델 생성
        self.model = create_model(model=model_name, input_length=input_length, output_dim=num_classes)
        
        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics (Val)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        
        # Metrics (Test)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        # ⚡ 교수님 요청 분석용 리스트 (Confidence 저장)
        self.test_preds = []
        self.test_targets = []
        self.test_confidences = [] 

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
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # ⚡ [핵심] Softmax로 확률(Confidence) 계산
        probs = torch.softmax(logits, dim=1) 
        confidence, preds = torch.max(probs, dim=1) # 가장 높은 확률값과 인덱스
        
        targets = torch.argmax(y, dim=1)
        loss = self.criterion(logits, y)

        self.test_acc(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)
        
        # 분석용 데이터 저장 (CPU로 이동)
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        self.test_confidences.append(confidence.detach().cpu())

        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1)

    def on_test_epoch_end(self):
        if not self.test_preds: return
        
        # 리스트 합치기
        preds = torch.cat(self.test_preds).numpy()
        targets = torch.cat(self.test_targets).numpy()
        confidences = torch.cat(self.test_confidences).numpy()
        
        # ⚡ [분석 1] CSV 저장 (나중에 회귀 오차와 비교용)
        os.makedirs("./result_analysis", exist_ok=True)
        timestamp = datetime.now().strftime('%m%d_%H%M%S')
        
        df = pd.DataFrame({
            'Target': targets,
            'Prediction': preds,
            'Confidence': confidences,
            'Correct': (targets == preds)
        })
        csv_path = os.path.join("./result_analysis", f"reliability_{self.hparams.model_name}_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n📄 신뢰도 분석 데이터 저장됨: {csv_path}")

        # ⚡ [분석 2] 신뢰도 히스토그램 (정답 vs 오답 분포)
        plt.figure(figsize=(10, 5))
        sns.histplot(data=df, x='Confidence', hue='Correct', bins=20, multiple="stack", kde=True)
        plt.title(f"Confidence Distribution (Correct vs Wrong)")
        plt.xlabel("Confidence Score")
        hist_path = os.path.join("./result_analysis", f"conf_hist_{self.hparams.model_name}_{timestamp}.png")
        plt.savefig(hist_path, dpi=300)
        plt.close()

        # ⚡ [분석 3] Confusion Matrix
        cm = confusion_matrix(targets, preds)
        class_names = ["acetone", "benzene", "toluene"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix ({self.hparams.model_name})")
        cm_path = os.path.join("./result_analysis", f"cm_{self.hparams.model_name}_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()
        
        self.test_preds.clear()
        self.test_targets.clear()
        self.test_confidences.clear()

    # ⚡ [빠진 부분 추가] 이게 없어서 에러가 났었습니다!
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)