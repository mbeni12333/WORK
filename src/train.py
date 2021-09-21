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

from torchsummary import summary

import copy # to create deep copies
import pickle as pk # for serialization

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.callbacks import TrainingDataMonitor
from pytorch_lightning.callbacks import LearningRateMonitor

import argparse
import numpy as np
import pandas as pd
import cv2
import os

from mil.data import CamelyonDatasetMIL
from mil.model import ChowderNT
from sklearn.neighbors import KernelDensity
from matplotlib.colors import LinearSegmentedColormap

def plotMap(GT, labels, coords):
    """
    """
    coords = (coords - coords.min(0))/(coords.ptp(0))
    
    coords_labels = coords[labels == 1, :]
    coords_gt = coords[GT == 1, :]

    kde_labels = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(coords_labels)
    kde_gt = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(coords_gt)

    # Regular grid to evaluate kde upon
    n_grid_points = 128
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
    plt.scatter(coords[:, 0], coords[:, 1], alpha=1, s=2)
    plt.contourf(xg, yg, np.exp(zz_gt), cmap=map_object, alpha=0.4)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.subplot(122)
    plt.title("Predicted")
    plt.scatter(coords[:, 0], coords[:, 1], alpha=1, s=2)
    plt.contourf(xg, yg, np.exp(zz_labels), cmap=map_object, alpha=0.4)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()  


def train(kwargs):
	lr_monitor = LearningRateMonitor(logging_interval='step')

	early_stopping_callback = EarlyStopping(monitor='train_loss')

	checkpoint_callback = ModelCheckpoint(
	monitor='train_loss',
	dirpath='checkpoints',
	save_top_k=3,
	filename='chowder_{epoch:02d}-val_loss{valid_loss:.4f}',
	auto_insert_metric_name=False
	)

	datamonitor = TrainingDataMonitor()

	callbacks=[lr_monitor, checkpoint_callback]


	train_dataloader = torch.utils.data.DataLoader(CamelyonDatasetMIL(root_dir="encoded_data_train_resnet18/", subset=0),
											batch_size=1, shuffle=True, num_workers=4)

	validation_dataloader = torch.utils.data.DataLoader(CamelyonDatasetMIL(root_dir="encoded_data_test_resnet18/",subset=1),
											batch_size=1, shuffle=False, num_workers=4)
												
	test_dataloader = torch.utils.data.DataLoader(CamelyonDatasetMIL(root_dir="encoded_data_test_resnet18/",subset=0),
											batch_size=1, shuffle=False, num_workers=4)



	model = ChowderNT(embd_size=512, loss_weight=torch.tensor(0.3))
	logger = TensorBoardLogger(os.path.abspath("lightning_logs"), log_graph=True, default_hp_metric=False, name="CHOWDER")
	trainer = pl.Trainer(gpus=1, max_epochs=100, auto_lr_find=True, accumulate_grad_batches=10, callbacks=callbacks, logger=logger)


	trainer.fit(model, train_dataloaders=train_dataloader, 
				val_dataloaders=validation_dataloader)
	trainer.test(model, dataloaders=test_dataloader)


	cm = metrics.confusion_matrix(y, yhat>0.5)
	print(cm/cm.sum())
	print(metrics.classification_report(y, yhat>0.5))


	fpr, tpr, thresholds = metrics.roc_curve(y, yhat)
	roc_auc = metrics.auc(fpr, tpr)
	display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='estimator')
	fig = plt.figure(figsize=(10, 10))
	ax = fig.gca()

	display.plot(ax) 

	plt.plot([0, 1], [0, 1], '--', linewidth=2)
	plt.show()

	colors = np.array(['gray', 'red'])
	bins = 200

	tumor_scores = []
	normal_scores = []

	for i in range(1, len(local_labels)):
		
		sorted_idx = np.argsort(local_scores[i])
		local_labels_sorted = local_labels[i][sorted_idx] 
		tumor = local_labels[i].sum()!=0
		
		normal_scores = np.concatenate((normal_scores, local_scores[i][np.bitwise_not(local_labels[i])]) )
		
		if(not tumor):
			continue
		
		tumor_scores = np.concatenate((tumor_scores, local_scores[i][local_labels[i]]))
		
		
	plt.figure(figsize=(10, 10))

	plt.hist(normal_scores, bins=bins, density=True, alpha=0.5, label="Normal scores")
	plt.hist(tumor_scores, bins=bins, density=True, alpha=0.5, label="Tumor scores")
	plt.legend()
	plt.show()


	local_labels = []
	local_scores = []
	coords = []

	for img, g, labels, xy in validation_dataloader:
		scores = model.embedding(img).view(-1).detach().cpu().numpy()
		local_scores.append(scores)
		local_labels.append(labels.view(-1).numpy())
		coords.append(xy)
		
		if g == 1:
			plotMap(labels.view(-1).numpy(), scores<-0., xy[0].cpu().numpy())



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train a MIL model")

	parser.add_argument("-d", "--dataset", type=str,
						default="weights/encoder", help="Encodded dataset path")

	parser.add_argument("-E", "--epochs",
						default=10, type=int, help="Maximum number of epochs")

	parser.add_argument("-a", "--acumulateGrad",
						default=8, type=int, help="Accumulate gradient batches")

	parser.add_argument("-b", "--batchSize",
						default=1, type=int, help="BatchSize used ")


	args = parser.parse_args()

	train(args)
