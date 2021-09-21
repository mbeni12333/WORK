import os
import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
import numpy as np # linear algebra library
import cv2 # for image processing methods 
import matplotlib.pyplot as plt # visualisation tool
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

import pandas as pd
import os # for managing file paths and os operations
import sys # system specefic calls
import time 
import pdb # debug

from sklearn.metrics import confusion_matrix, classification_report

import torch # Deep learning framework
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.neighbors import KernelDensity

import torchvision # for creating datasets, and pretrained models
from torch.utils.tensorboard import SummaryWriter # visualise the learning
from  torch.utils.data import Dataset, DataLoader # parellel dataset loader
from torchvision import models, datasets, transforms, utils

from torchsummary import summary

import copy # to create deep copies
import pickle as pk # for serialization
import tqdm
import glob

from torch.optim.lr_scheduler import ExponentialLR

from data import *


class ChowderNT(pl.LightningModule):
    """
    """
    
    def __init__(self, J=1, R=10, L=0, Tau=0.5, loss_weight=1.0, embd_size=512, learning_rate=0.01):
        super().__init__()
        
        self.R = R
        self.Tau = Tau
        self.learning_rate = learning_rate
        self.example_input_array = torch.zeros((1, embd_size, 1300))
        print(self.example_input_array)
        self.embedding = nn.Conv1d(embd_size, 1, 1)
        
        
        self.params = {"embd_size":embd_size, "loss_weight": loss_weight, "learning_rate": learning_rate}
        
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight = loss_weight)
        
        mlp1 = nn.Linear(2*R, 200)
        mlp2 = nn.Linear(200, 100)
        mlp3 = nn.Linear(100, 1)
        
        self.mlp = nn.Sequential(mlp1,nn.Sigmoid(),nn.Dropout(0.2),
                                 mlp2,nn.Sigmoid(),nn.Dropout(0.2),
                                 mlp3)
        
        self.init_parameters()
    
    def init_parameters(self):
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
        
        self.mlp.apply(init_layer)
        self.embedding.apply(init_layer)
    
    def predictPatchesLevel(self, X, tau=0.8):
        """
        """
        X = self.embedding(X).view(X.shape[0], -1).cpu().numpy()
        threshold = np.quantile(X, tau)
        labels = X > threshold
    
        return labels
        
        
    def forward(self, X):
        """ """
        #print(X.shape)
        X = self.embedding(X).view(X.shape[0], -1)
        #print(X.shape)
        top, _ = torch.topk(X, self.R, 1, largest=True, sorted=True)
        low, _ = torch.topk(X, self.R, 1, largest=False, sorted=True)
        
        
        X = torch.cat((top, low), 1)
        #print(X.shape, top.shape, low.shape)
        X = self.mlp(X)
        return X
    
    def validation_step(self, batch, batch_idx):
        x, y, Y = batch
        y_hat = self.forward(x)
        
        if y == 1:
            self.logger.experiment.add_histogram("embedding/tumor", self.embedding(x), batch_idx)
        else:
            self.logger.experiment.add_histogram("embedding/normal", self.embedding(x), batch_idx)
        #print(y_hat.shape, y.shape)
        loss = self.criterion(y_hat, y.reshape(-1, 1))
        # Logging to TensorBoard by default
        self.log('valid_loss', loss)
        return loss    
    
    def test_step(self, batch, batch_idx):
        x, y, Y = batch
        y_hat = self.forward(x)
        #print(y_hat.shape, y.shape)
        loss = self.criterion(y_hat, y.reshape(-1, 1))
        # Logging to TensorBoard by default
        self.log('test_loss', loss)
        
        return loss
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, Y = batch
        y_hat = self.forward(x)
        #print(y_hat.shape, y.shape)
        loss = self.criterion(y_hat, y.reshape(-1, 1))
        
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss
      
    def training_epoch_end(self, _):
        
        self.logger.experiment.add_histogram("weights/embd", self.embedding.weight, self.trainer.global_step)
        self.logger.experiment.add_histogram("weights/mlp1", self.mlp[0].weight, self.trainer.global_step)
        self.logger.experiment.add_histogram("weights/mlp2", self.mlp[3].weight, self.trainer.global_step)
        self.logger.experiment.add_histogram("weights/mlp3", self.mlp[6].weight, self.trainer.global_step)
    
    #def validation_epoch_end(self, validation_step_outputs):
    #	self.i = self.i+1
    #    self.logger.experiement.add_pr_curve()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params':self.embedding.parameters(),'weight_decay':0.1},
            {'params':self.mlp.parameters()}
        ], lr=self.learning_rate)
        
        scheduler = ExponentialLR(optimizer, 0.96)

        return [optimizer], [scheduler] 
