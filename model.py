import torch
import torch.nn as nn
from torch import nn, optim
from pytorch_lightning import seed_everything, LightningModule, Trainer
from BCM_dataset import BCMDataset
from torch.utils.data import DataLoader

# Based on: https://towardsdatascience.com/pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-7892f3ba83c0


class LSTM_Model(LightningModule):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        
        # Network layers        
        self.lstm = nn.LSTM(input_size = 16,
                            hidden_size = 64,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = True)
        self.fc = nn.Linear(128, 5)
        self.output = nn.Sigmoid()
        
        self.lr = 0.000001
        self.batch_size = 1
        
    def forward(self, x):
        lstm_out, (ht, ct) = self.lstm(x)
        x = self.fc(torch.flatten(ht))
        return self.output(x)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        loader = DataLoader(BCMDataset('data/mfcc_array2.npy'), batch_size=self.batch_size, shuffle=False)
        return loader
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = nn.CrossEntropyLoss(self(x), y)
        return {'loss': loss, 'log': {'train_loss': loss}}
    
    def val_dataloader(self):
        loader = DataLoader(BCMDataset('data/mfcc_array2.npy'), batch_size=self.batch_size, shuffle=False)
        return loader
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = nn.CrossEntropyLoss(self(x), y)
        return {'val_loss': loss, 'log': {'val_loss': loss}}
    
    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_acc in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 
                   'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        print("OUR LR:",self.lr)
        return results
    
    