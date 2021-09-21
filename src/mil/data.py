import os
import torch
import pandas as pd
from skimage import io, transform
import seaborn as sns
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

import glob

import copy # to create deep copies
import pickle as pk # for serialization
import tqdm


idx = np.arange(269, dtype=int)
np.random.shuffle(idx)

train_idx = idx[:int(len(idx)*0.8)]
val_idx = idx[int(len(idx)*0.8): int(len(idx)*0.9)]
test_idx = idx[int(len(idx)*0.9):]

subsets = [train_idx, val_idx, test_idx]


class CamelyonDatasetMIL(Dataset):
    """
    Patch Camelyon dataset for Multiple instance learning 
    
    """

    def __init__(self, root_dir="encoded_data_resnet18",
    		  subset=0,
                 transform=None):
        """
        """

        self.root_dir = root_dir
        self.transform = transform
        # load all the dataset to the system ram

        files = np.array(sorted(glob.glob(f"../{root_dir}/*.npz")))

        files  = np.array(files[subsets[subset]])
        
        self.X = []
        self.Y = []
        self.coords = []
        
        for file in tqdm.tqdm(files):
            with np.load(file, "r") as f:
                self.X.append(f["imgs"].transpose())
                self.Y.append(f["labels"])
                self.coords.append(f["xy"])
        	
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = self.X[idx]
        labels = self.Y[idx]
        
        global_label = float(labels.sum() >= 1)
    	
        return imgs, global_label, labels, self.coords[idx]
        
        
    def plotMap(self, idx, labels):
        """
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        GT = self.Y[idx]
        coords = self.coords[idx]

        coords_labels = coords[labels == 1]
        coords_gt = coords[GT == 1]
        
        
        kde_labels = KernelDensity(kernel='gaussian', bandwidth=1000).fit(coords_labels)
        kde_gt = KernelDensity(kernel='gaussian', bandwidth=1000).fit(coords_gt)
        
        # Regular grid to evaluate kde upon
        n_grid_points = 500
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)

        xx = np.linspace(xmin - 0.5, xmax + 0.5, n_grid_points)
        yy = np.linspace(ymin - 0.5, ymax + 0.5, n_grid_points)

        xg, yg = np.meshgrid(xx, yy)
        grid_coords = np.c_[xg.ravel(), yg.ravel()]

        zz_labels = kde_labels.score_samples(grid_coords) # Evaluate density on grid points
        zz_labels = zz_labels.reshape(*xg.shape)

        zz_gt = kde_gt.score_samples(grid_coords) # Evaluate density on grid points
        zz_gt = zz_gt.reshape(*xg.shape)

        ncolors = 256
        color_array = plt.get_cmap('jet')(range(ncolors))

        # change alpha values
        color_array[:,-1] = np.linspace(1.0,0.0,ncolors)
        map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
        
        fig = plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.title("Ground Truth")
        plt.scatter(coords[:, 0], coords[:, 1], alpha=1, s=10)
        plt.contourf(xx, yy, np.exp(zz_gt), cmap=map_object, alpha=0.4)
        plt.subplot(122)
        plt.title("Predicted")
        plt.scatter(coords[:, 0], coords[:, 1], alpha=1, s=10)
        plt.contourf(xx, yy, np.exp(zz_labels), cmap=map_object, alpha=0.4)
        plt.show()     
