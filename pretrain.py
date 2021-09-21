from pytorch_lightning.callbacks import early_stopping, model_checkpoint, ProgressBar, LearningRateMonitor
from pl_bolts.models.self_supervised import Moco_v2, BYOL
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pl_bolts.datamodules import AsynchronousLoader
from pytorch_lightning.loggers import TensorBoardLogger

import argparse
import numpy as np
import pandas as pd
import cv2
import os

from backbone.data import Camelyon16Preprocessed
from backbone.model import Mocov2_gpu_transform

device = "cuda" if torch.cuda.is_available() else "cpu"

def pretrain(kwargs):
    
    checkpoint_callback = model_checkpoint.ModelCheckpoint(dirpath=kwargs["dirpath"],
                                                       monitor="train_loss")
    earlystop_callback = early_stopping.EarlyStopping(monitor="train_loss")
    lrmonitor_callback = LearningRateMonitor(logging_interval='step')
    datamodule = Camelyon16Preprocessed()

    model = Mocov2_gpu_transform(kwargs["backbone"],
                    embd_dim=kwargs["embdSize"],
                    num_negatives=kwargs["numNegatives"],
                    use_mlp=True,
                    batch_size=kwargs["batchSize"],
                    learning_rate=kwargs["learningRate"])


    logger = TensorBoardLogger(kwargs["logdir"], default_hp_metric=False, name=kwargs["name"], log_graph=True)

    trainer = pl.trainer.Trainer(gpus=1, callbacks=[checkpoint_callback],
                                max_epochs=kwargs["epochs"],resume_from_checkpoint=kwargs["resume"],
                                precision=16,accumulate_grad_batches=kwargs["accumulateGrad"],
                                logger=logger)

    trainer.fit(model,datamodule=datamodule)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pretrain a Backbone using a self supervised learning method")

    parser.add_argument("-d", "--dirpath", type=str,
                        default="weights/encoder", help="Location of checkpoints")

    parser.add_argument("-b", "--backbone",
                        default="resnet50", type=str, help="Backbone used")
    
    parser.add_argument("-E", "--epochs",
                        default=10, type=int, help="Maximum number of epochs")

    parser.add_argument("-r", "--resume",
                        default=None, type=str, help="Resume training from a checkpoint")
    
    parser.add_argument("-a", "--acumulateGrad",
                        default=8, type=int, help="Accumulate gradient batches")
    
    parser.add_argument("-n", "--name",
                        default=None, type=str, help="Name of the experience")

    parser.add_argument("-e", "--embdSize",
                        default=128, type=int, help="Embedding size used in training phase")
    
    parser.add_argument("-b", "--batchSize",
                        default=256, type=int, help="Batchsize used for training")
    

    parser.add_argument("-N", "--numNegatives",
                        default=65536, type=int, help="Number of negatives (size of the queue)")
    
    parser.add_argument("-l", "--learningRate",
                        default=0.03, type=float, help="Learning rate used ")

    parser.add_argument("-L", "--logdir",
                        default="lightning_logs", type=str, help="Lightning log folder for tensorboard visualisation")

    args = parser.parse_args()
    
    pretrain(args)