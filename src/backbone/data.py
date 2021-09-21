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


class Camelyon16PreprocessedDataset(torch.utils.data.Dataset):
    """
    Dataset of unlabelled patches 
    """
    
    def __init__(self, data, transforms=None):
        """
        data: pandas dataframe
        """
        self.data = data
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch = self.data.iloc[idx]
        img = read_image(patch["path"], mode=ImageReadMode.RGB)/255
        label = patch["local_class"]
        
        return (img, img), label


class Camelyon16Preprocessed(pl.LightningDataModule):
    """

    """
    
    def __init__(self, data_path="processed_data", valid_portion=0.2, warmup=1):
        super().__init__()
        self.data_path = data_path
        self.data = pd.read_csv(os.path.join(self.data_path, "data.csv"))
        idx = np.arange(len(self.data))
        np.random.shuffle(idx)
        idx = idx[:int(len(idx)*warmup)]
        self.train_idx = idx[int(len(idx)*valid_portion):]
        self.valid_idx = idx[:int(len(idx)*valid_portion)]
        

    def vale_dataloader(self):
    
        dataset = Camelyon16PreprocessedDataset(self.data.iloc[self.valid_idx],
                                       transforms=[transforms.ToTensor(),transforms.ToTensor()])
            
        dataloader = AsynchronousLoader(torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=128,
                                         prefetch_factor=4,
                                         num_workers=8,
                                         pin_memory=True,
                                         drop_last=True), device=device)
        
        return dataloader
    
    def train_dataloader(self):
        dataset = Camelyon16PreprocessedDataset(self.data.iloc[self.train_idx],
                                       transforms=[transforms.ToTensor(),transforms.ToTensor()])
                                        
        
        dataloader = AsynchronousLoader(torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=256,
                                         shuffle=True,
                                         prefetch_factor=8,
                                         num_workers=8,
                                         pin_memory=True,
                                         drop_last=True), device=device)
        return dataloader


class Camelyon16PreprocesseSlidedDataset(torch.utils.data.Dataset):
    """
    Dataset of unlabelled patches
    """
    
    def __init__(self, csv_file, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.data["slide"] = self.data["path"].str.split("/", expand=True)[1]
        groups = self.data.groupby("slide")
        self.slidenames = np.unique(self.data["slide"])
        self.patchesByslide = list(groups.groups.values())
        self.transforms = transforms
        
    def __len__(self):
        return len(self.patchesByslide)

    def __getitem__(self, idx):
        patches = self.data.iloc[self.patchesByslide[idx]]

        imgs = torch.stack([read_image(path, ImageReadMode.RGB) for path in patches["path"]])/255

        global_label = patches["global_class"].iloc[0]
        local_labels = patches["local_class"].to_numpy()
        
        imgs = self.transforms(imgs)
        
        return imgs, (global_label, local_labels)
    
    def getpatchesidx(self, idx):
        return self.data.iloc[self.patchesByslide[idx]], self.slidenames[idx], self.patchesByslide[idx]
    
        
    def getbatchitem(self, patches, ids, batch_idx=0, batch_size=256):

        if(len(patches)//batch_size < batch_idx):
            print(batch_idx)
            return None
        
        patches = patches.iloc[batch_idx*batch_size:(batch_idx+1)*batch_size]
        imgs = torch.stack([read_image(path, ImageReadMode.RGB) for path in patches["path"]])/255
        global_label = patches["global_class"].iloc[0]
        local_labels = patches["local_class"].to_numpy()
        
        imgs = self.transforms(imgs)

        return (imgs, (global_label, local_labels), ids[batch_idx*batch_size:(batch_idx+1)*batch_size])
    
