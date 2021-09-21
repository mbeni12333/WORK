import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import json
import os
import pytorch_lightning as pl
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid 
from torchvision import models
from PIL import Image
from pytorch_lightning.callbacks import early_stopping, model_checkpoint, ProgressBar
from pl_bolts.models.self_supervised import Moco_v2, BYOL
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
import tqdm
from sklearn.decomposition import PCA
import gc
from torchvision.io import ImageReadMode, read_image
import torchvision
import threading, queue
from concurrent.futures import ThreadPoolExecutor

import argparse
import numpy as np
import pandas as pd
import cv2
import os

from backbone.data import Camelyon16PreprocesseSlidedDataset
from backbone.model import *

#We would like to visualize the latent space using different Encoders
gc.collect()
torch.cuda.empty_cache()


def encode(model, idx, dataset, device="cuda", save_encoding=None, batch_size=512, num_workers=4, emd_size=512):
    
    patches, slidename, ids = dataset.getpatchesidx(idx)

    encodded = []
    labels = []
    xys = []
    
    available_batches = queue.Queue()
    prefetched = queue.Queue(maxsize=4)
    
    [available_batches.put(i) for i in range(len(patches)//batch_size)]
    
    def worker():
        while not available_batches.empty():
            item = available_batches.get()
            prefetched.put(dataset.getbatchitem(patches, ids, item, batch_size))
            available_batches.task_done() 

    with ThreadPoolExecutor() as executor:
        [executor.submit(worker) for i in range(num_workers)]
 
    
        for i in tqdm.tqdm(range(len(patches)//batch_size)):
            imgs, (global_label, local_labels), indicies = prefetched.get()
            xy = dataset.data.iloc[indicies][["x", "y"]].values
            imgs = imgs.to(device)
            encodded.append(model(imgs).detach().cpu().numpy().reshape(-1, emd_size))
            labels.append(local_labels)
            xys.append(xy)
        
        available_batches.join()
        
        encodded=np.concatenate(encodded,0)
        labels=np.concatenate(labels, 0)
        xys = np.concatenate(xys, 0)
    
        if save_encoding is not None:
            with open(os.path.join(save_encoding, f"{slidename}.npz"), "wb") as f:

                np.savez_compressed(f, imgs=encodded, labels=labels, xy=xys)

def encode_dataset(kwargs):

    data_transforms = transforms.Compose([imagenet_normalization()])
    dataset = Camelyon16PreprocesseSlidedDataset(kwargs["dataset"]+"/data.csv", data_transforms)

    # model
    model = models.resnet18(True)
    model = nn.Sequential(*list(model.children())[:-1]).cpu().eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    HOME = os.getcwd()
    save_encoding=os.path.join(HOME, kwargs["distPath"])
    batch_size=kwargs["batchSize"]
    num_workers=kwargs["numWorkers"]

    for idx in range(len(dataset)):
        encode(model, idx, dataset, device=device, save_encoding=save_encoding, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Encode a dataset into numpy arrays feature vectors")

    parser.add_argument("-d", "--dataset", type=str,
                        default="processed_data", help="Dataset root folder")

    parser.add_argument("-m", "--model", type=str,
                        default="resnet18", help="Model used for the encoding")

    parser.add_argument("-p", "--distPath",
                        default="encodded_data_train_imagenet", type=str, help="Destination path for the encodded dataset")
       
    parser.add_argument("-n", "--numWorkers",
                        default=8, type=int, help="number of workers to use for loading images")
    
    parser.add_argument("-b", "--batchSize",
                        default=256, type=int, help="Batchsize used for training")
       

    args = parser.parse_args()
    
    encode_dataset(args)