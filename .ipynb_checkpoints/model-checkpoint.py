import torch
import torch.nn as nn
from torch import nn, optim
from pytorch_lightning import seed_everything, LightningModule, Trainer
from BCM_dataset import BCMDataset
from torch.utils.data import DataLoader
from BCM_dataset import concat_train_test_datasets
from torchmetrics import Accuracy

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
        
        self.lr = 0.0001
        self.batch_size = 128
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        
        #Create the datasets
        self.train_set, self.val_set = concat_train_test_datasets('data')
        
        
                
    def forward(self, x):
        lstm_out, (ht, ct) = self.lstm(x[:,None,:]) #Insert empty dimension for batch size
        # Assert that there are only 2 layers
        assert ht.shape[0] == 2, "ht.shape[0] != 2. To use more than 2 layers, change this code"
        
        ht = torch.hstack((ht[0],ht[1])) #Concatenate the two directions. todo: accommodate any number of layers
         
        x = self.fc(ht)
        return self.output(x)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        
        loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)
        return loader
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.loss(self(x), y)
        preds = self(x)
        self.accuracy(torch.argmax(preds), torch.argmax(y))
        self.log('train_acc_step', self.accuracy)
        self.log("train_loss", loss)
        return {'train_loss': loss, 'log': {'train_loss': loss}}
    
    def val_dataloader(self):
        loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
        return loader
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("val_loss",loss)
        return {'val_loss': loss, 'log': {'val_loss': loss}}
    
    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        
        
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 
                   'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        print(results)
    
        return results
    
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy)
        print(f'Accuracy: {self.accuracy}')
    
