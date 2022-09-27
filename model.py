import torch
import torch.nn as nn
from torch import nn, optim
from pytorch_lightning import seed_everything, LightningModule, Trainer
from BCM_dataset_v2 import bcmDataset, concat_train_test_datasets
from torch.utils.data import DataLoader

from torchmetrics import Accuracy

# Based on: https://towardsdatascience.com/pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-7892f3ba83c0


class LSTM_Model(LightningModule):
    def __init__(self, file_path, window_size = 3, stride = 0.032, MFCC_stride = 0.032):
        super(LSTM_Model, self).__init__()
        
        self.file_path = file_path
        self.window_size = window_size
        self.stride = stride
        self.MFCC_stride = MFCC_stride
        self.mfccs_pr_window = int(window_size/MFCC_stride)
        
        # Network layers        
        self.lstm = nn.LSTM(input_size = 16,
                            hidden_size = 64,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = True)
        self.fc = nn.Linear(128 , 5)
        self.flatten = nn.Flatten()
        self.output = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1) # Dim 1 when using 1 output pr window
        self.lr = 0.001
        self.batch_size = 32
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        
        #Create the datasets
        self.train_set, self.val_set = concat_train_test_datasets(file_path, window_size = window_size, stride = self.stride, MFCC_stride = MFCC_stride)
        
        
                
    def forward(self, x):
        lstm_out, (ht, ct) = self.lstm(x) 
        ht = ht.type_as(x)
        # Flatten the final hidden states from the LSTM layers. The axes are moved using moveaxis, to the the batch dimension first
        x = self.flatten(torch.moveaxis(ht,0,1)) # This version of the flatten layer starts from dim 1, which avoids flattening the batch dimension
        x = self.fc(x)
        x = self.output(x)
        #view as (batch_size, mfccs_pr_window, 5)
        #x = x.view(-1, self.mfccs_pr_window, 5) # No longer needed
        x = self.sm(x) # Softmax to get probabilities
        return x
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        
        loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers = 8)
        return loader
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y) # Flatten to collect batch dimension and mfccs_pr_window dimension
                
        
        # Use argmax to get the index of the highest value in the output vector along the 2. dimension (the 5 classes)
        # Then use flatten to get a 1D tensor with every guess
        self.accuracy(preds.argmax(1), y.argmax(1)) # Argmax finds the index of the highest value in the output vector along the 2. dimension (dim 1) (the 5 classes)
        
        #Log the loss and accuracy
        self.log('train_acc_step', self.accuracy)
        self.log("train_loss", loss)
        return {'loss': loss, 'log': {'train_loss': loss}}
    
    def val_dataloader(self):
        loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers = 8)
        return loader
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("val_loss",loss)
        
        preds = self(x)                
        
        # Use argmax to get the index of the highest value in the output vector along the 2. dimension (the 5 classes)
        # Then use flatten to get a 1D tensor with every guess
        self.val_accuracy(preds.argmax(1), y.argmax(1)) 
        
        #Log the loss and accuracy
        self.log('val_acc_step', self.val_accuracy)
        self.log("val_loss", loss)        
        
        
        return {'val_loss': loss, 'log': {'val_loss': loss}}
    
    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        
        
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 
                   'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        print(results)
        
        self.log('val_acc_epoch', self.val_accuracy)
        print(f'Accuracy: {self.val_accuracy.compute()}')
        
        return results
    
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy)
        print(f'Accuracy: {self.accuracy.compute()}')
    
    
if False: #for testing. Debugging doesn't work with separate files in this case??

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from pytorch_lightning import seed_everything, LightningModule, Trainer
    from torch import save
    from pytorch_lightning.callbacks import EarlyStopping

    torch.manual_seed(1)
    
    seed_everything(42)
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #Check for cuda 
    #device = 'cpu'
    print(f'Using {device} device')

    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=True, mode='min')

    model = LSTM_Model('data/bcm/').to(device)
    #trainer = Trainer(max_epochs=100, min_epochs=1, auto_lr_find=False, auto_scale_batch_size=False, callbacks=[early_stop_callback],enable_checkpointing=False)
    trainer = Trainer(max_epochs=3, min_epochs=1, auto_lr_find=False, auto_scale_batch_size=False,enable_checkpointing=False)
    trainer.tune(model)

    trainer.fit(model)
    #save(model.state_dict(), '/trained_model')