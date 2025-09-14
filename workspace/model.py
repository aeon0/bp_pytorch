import torch
from pytorch_lightning import LightningModule
from torch import nn


class SimpleClassifier(LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss, prog_bar=True)

    def predict_step(self, batch):
        x, y = batch
        logits = self(x)
        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
