import openslide as ops
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import cv2
from openslide.deepzoom import DeepZoomGenerator
import os
import wsiprocess as wp
import pymeanshift as pms
from PIL import Image
import glob
import gc

# class Slide(wp.slide):
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#     def crop(self, x, y, w, h):
#         if self.backend == "openslide":
#             img = cv2.resize(np.array(self.slide.read_region((x, y), 0, (w, h))), (224, int(h/w * 224)))
#             return Image.fromarray(img)

def plot_tissue(filepath="tumor/tumor_011.tiff", xmlfilepath=None, size=1000, plot=False):

    slide = wp.slide(filepath)
    
    print("Reading annotations ...")
    annotation = wp.annotation(xmlfilepath)
    annotation.make_masks(slide, size=size, foreground=False)
    
    thumb = np.array(slide.get_thumbnail(size))

    print("segmenting the tissue ...")
    thumb_hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    segmented = cv2.pyrMeanShiftFiltering(thumb_hsv, 7, 30)

    notcross= segmented[:, :, 2] >= 60
    notbackground= segmented[:, :, 1] >= 40 # 
    mask = notbackground*notcross
      

    print("Making masks ...")
    tumor_mask = 0

    
    for key in annotation.masks.keys():
        if key in ["_0", "_1"]:
            tumor_mask |= cv2.imread(f"tmp/{key}_thumb.png").max(2)
    
    tumor_mask = (tumor_mask>0)*1
    h, w = slide.height, slide.width
        
    annotation.masks["foreground"] = cv2.resize(mask.astype(np.uint8), [w, h], interpolation=cv2.INTER_NEAREST)
#     annotation.masks["tumor"] = 0
#     if("_0" in annotation.masks.keys()):
#         annotation.masks["tumor"] = annotation.masks["_0"]
#     if("_1" in annotation.masks.keys()):
#         annotation.masks["tumor"] += annotation.masks["_1"]
    annotation.masks["tumor"] = cv2.resize(tumor_mask.astype(np.uint8), [w, h], interpolation=cv2.INTER_NEAREST)
    annotation.masks["normal"] = annotation.masks["foreground"] - annotation.masks["tumor"]
    
    annotation.dot_bbox_height = 1
    annotation.dot_bbox_width = 1
    
    if plot:
        plt.figure(figsize=(18, 22))
        plt.subplot(121)
        plt.title("original")
        plt.imshow(thumb)
        plt.subplot(122)
        plt.title("Masked tissue + tumor")
        plt.imshow(mask[:, :, None]*thumb)
        h, w = mask.shape
        plt.imshow(cv2.resize(tumor_mask.astype(np.uint8), [w, h]), alpha=0.3)
        plt.show()
    
    return slide, annotation

for file in sorted(glob.glob("training/tumor/*.tif")):
    print(file)
    
    print("Creating the annotations ...")
    slide, annotation = plot_tissue(file, file.split(".")[0]+".xml", 2000, plot=True)
    
    print("Creating the patches ...")
    patcher = wp.patcher(slide, "classification", annotation, patch_width=224, patch_height=224, on_foreground=0.001, on_annotation=0.01, save_to="processed2")
    path = os.path.basename(file).split(".")[0]
    annotation.export_thumb_masks(f"processed2/{path}", size=512)
    slide.export_thumbnail(f"processed2/{path}.jpg", size=512)
    patcher.get_patch_parallel(['tumor', 'normal'])
    
    del patcher
    del slide
    del annotation
    
    gc.collect()