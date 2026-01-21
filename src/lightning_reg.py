import torch
import torch.nn.functional as F
import lightning as L
from src.model import create_model

class GasRegModel(L.LightningModule):
    def __init__(self, model_name, input_length, output_dim=1, lr=1e-4, max_ppm=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model=model_name, input_length=input_length, output_dim=output_dim)

    def forward(self, x):
        return self.model(x)

    def _calculate_metrics(self, preds, y):
        preds, y = preds.view(-1), y.view(-1)
        loss = F.mse_loss(preds, y) 

        preds_ppm = preds * self.hparams.max_ppm
        y_ppm = y * self.hparams.max_ppm
        
        mask = y_ppm > 0.001
        if torch.sum(mask) > 0:
            diff = (y_ppm[mask] - preds_ppm[mask]) / y_ppm[mask]
            mape = torch.mean(torch.abs(diff)) * 100
            mspe = torch.mean(diff ** 2) * 100
        else:
            mape = torch.tensor(0.0, device=self.device)
            mspe = torch.tensor(0.0, device=self.device)
            
        return loss, mspe, mape 

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, mspe, mape = self._calculate_metrics(self(x), y.float())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_mape", mape, on_epoch=True, prog_bar=True)
        self.log("train_mspe", mspe, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, mspe, mape = self._calculate_metrics(self(x), y.float())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mape", mape, prog_bar=True)
        self.log("val_mspe", mspe, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, mspe, mape = self._calculate_metrics(self(x), y.float())
        self.log("test_loss", loss)
        self.log("test_mape", mape)
        self.log("test_mspe", mspe)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)