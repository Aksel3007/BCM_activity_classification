import torch
import torch.nn as nn
from torch import nn, optim
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.loggers import NeptuneLogger
from BCM_dataset_v2 import bcmDataset, concat_train_test_datasets
from torch.utils.data import DataLoader

from torchmetrics import Accuracy


class LSTM_Model(LightningModule):
    def __init__(self, 
                 file_path, 
                 window_size = 3, 
                 stride = 0.032, 
                 MFCC_stride = 0.032, 
                 lstm_layers = 1, 
                 lstm_hidden_size = 64, 
                 batch_size = 32, 
                 lr = 0.001,
                 weight_decay = 0):
        
        super(LSTM_Model, self).__init__()
        
        self.file_path = file_path
        self.window_size = window_size
        self.stride = stride
        self.MFCC_stride = MFCC_stride
        self.lstm_layers = lstm_layers
        self.hidden_size = lstm_hidden_size
        self.lr = lr
        self.batch_size = batch_size
        
        self.mfccs_pr_window = int(window_size/MFCC_stride)
        
        # Network layers        
        self.lstm = nn.LSTM(input_size = 16,
                            hidden_size = self.hidden_size,
                            num_layers = self.lstm_layers,
                            batch_first = True,
                            bidirectional = True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.hidden_size*2*self.lstm_layers , 5)
        self.output = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1) # Dim 1 when using 1 output pr window
        
        self.loss = nn.CrossEntropyLoss()
        
        self.accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        
        #Create the datasets
        self.train_set, self.val_set = concat_train_test_datasets(self.file_path, window_size = self.window_size, stride = self.stride, MFCC_stride = self.MFCC_stride)
        self.weight_decay = weight_decay
        
                
    def forward(self, x):
        lstm_out, (ht, ct) = self.lstm(x) 
        ht = ht.type_as(x)
        # Flatten the final hidden states from the LSTM layers. The axes are moved using moveaxis, to the the batch dimension first
        x = self.flatten(torch.moveaxis(ht,0,1)) # This version of the flatten layer starts from dim 1, which avoids flattening the batch dimension
        x = self.fc(x)
        x = self.output(x)
        x = self.sm(x) # Softmax to get probabilities
        return x
    
    def configure_optimizers(self):
        self.logger.experiment["metadata/file_path"].log(self.file_path)
        self.logger.experiment["metadata/window_size"].log(self.window_size)
        self.logger.experiment["metadata/stride"].log(self.stride)
        self.logger.experiment["metadata/MFCC_stride"].log(self.MFCC_stride)
        self.logger.experiment["metadata/mfccs_pr_window"].log(self.mfccs_pr_window)
        self.logger.experiment["metadata/lr"].log(self.lr)
        self.logger.experiment["metadata/batch_size"].log(self.batch_size)
        self.logger.experiment["metadata/lstm_layers"].log(self.lstm_layers)
        self.logger.experiment["metadata/lstm_hidden_size"].log(self.hidden_size)
        self.logger.experiment["metadata/weight_decay"].log(self.weight_decay)
        
        self.logger.experiment["metadata/train_set_length"].log(len(self.train_set))
        self.logger.experiment["metadata/val_set_length"].log(len(self.val_set))
        
        return optim.Adam(self.parameters(), lr=self.lr,  weight_decay = self.weight_decay)
    
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
    
    neptune_logger = NeptuneLogger(
        project="NTLAB/BCM-activity-classification",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYTA4NzcxMy1lYmQ2LTQ3NTctYjRhNC02Mzk1NjdjMWM0NmYifQ==",
        source_files=["train_model.ipynb", "model.py", "BCM_dataset_v2.py"]
    ) 


    model = LSTM_Model('data/bcm/', lstm_layers = 2).to(device)
    #trainer = Trainer(max_epochs=100, min_epochs=1, auto_lr_find=False, auto_scale_batch_size=False, callbacks=[early_stop_callback],enable_checkpointing=False)
    trainer = Trainer(max_epochs=3, min_epochs=1, auto_lr_find=False, auto_scale_batch_size=False,enable_checkpointing=False, logger = neptune_logger)
    trainer.tune(model)

    trainer.fit(model)
    #save(model.state_dict(), '/trained_model')