import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
import numpy as np # linear algebra library
import cv2 # for image processing methods 
import matplotlib.pyplot as plt # visualisation tool
from matplotlib import gridspec

import seaborn as sns
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

import torchvision # for creating datasets, and pretrained models
from torch.utils.tensorboard import SummaryWriter # visualise the learning
from  torch.utils.data import Dataset, DataLoader # parellel dataset loader
from torchvision import models, datasets, transforms, utils
from torchviz import make_dot
from torchsummary import summary

import copy # to create deep copies
import pickle as pk # for serialization

from pytorch_lightning.callbacks import early_stopping, model_checkpoint, ProgressBar, LearningRateMonitor
from pl_bolts.models.self_supervised import Moco_v2, BYOL
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pl_bolts.datamodules import AsynchronousLoader
from pytorch_lightning.loggers import TensorBoardLogger

from data import *

device = "cuda" if torch.cuda.is_available() else "cpu"

moco_transform = nn.Sequential(transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                                   transforms.GaussianBlur(23, sigma=(0.1, 2.0)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomGrayscale(),
                                   imagenet_normalization()).to(device).eval()

def getResnet18():
    MODEL_PATH = '../models/encoder/resnet18.ckpt'
    RETURN_PREACTIVATION = True  # return features from the model, if false return classification logits
    NUM_CLASSES = 2  # only used if RETURN_PREACTIVATION = False


    def load_model_weights(model, weights):

        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print('No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)

        return model


    model = torchvision.models.__dict__['resnet18'](pretrained=False)

    state = torch.load(MODEL_PATH, map_location='cuda:0')

    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    model = load_model_weights(model, state_dict)

    if RETURN_PREACTIVATION:
        model.fc = torch.nn.Sequential()
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model

class Mocov2_gpu_transform(Moco_v2):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        zero_img = torch.FloatTensor(np.zeros((1, 3, 224, 224)))
        zero_queue = torch.FloatTensor(np.zeros((128, 8192)))
        self.example_input_array = (zero_img, zero_img, zero_queue)
        
    def training_step(self, batch, batch_idx):
        
        (img_q, img_k), label = batch
        
        with torch.no_grad():
            img_q = moco_transform(img_q)
            img_k = moco_transform(img_k)

        batch = (img_q, img_k), label

        return super().training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        
        (img_q, img_k), label = batch
        
        with torch.no_grad():
            img_q = moco_transform(img_q)
            img_k = moco_transform(img_k)

        batch = (img_q, img_k), label
        
        return super().validation_step(batch, batch_idx)

    
class Byol_gpu_transform(BYOL):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        zero_img = torch.FloatTensor(np.zeros((1, 3, 224, 224)))
        self.example_input_array = zero_img
        
    def training_step(self, batch, batch_idx):
        
        (img_q, img_k), label = batch
        
        with torch.no_grad():
            img_q = moco_transform(img_q)
            img_k = moco_transform(img_k)

        batch = (img_q, img_k), label

        return super().training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        
        (img_q, img_k), label = batch
        
        with torch.no_grad():
            img_q = moco_transform(img_q)
            img_k = moco_transform(img_k)

        batch = (img_q, img_k), label
        
        return super().validation_step(batch, batch_idx)
