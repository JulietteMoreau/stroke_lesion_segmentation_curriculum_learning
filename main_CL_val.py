#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:25:22 2022

@author: moreau
"""

import matplotlib.pyplot as plt
import os
import glob

# torch stuff
import torch
torch.cuda.empty_cache()
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# torchsummary and torchvision
from torchsummary import summary
from torchvision.utils import save_image

# matplotlib stuff
import matplotlib.pyplot as plt
import matplotlib.image as img

# numpy and pandas
import numpy as np
import pandas as pd
import random as rd

# Common python packages
import datetime
import os
import sys
import time

#monai stuff
from monai.transforms import RandSpatialCropSamplesD,SqueezeDimd, SplitChannelD,RandWeightedCropd,\
    LoadImageD, EnsureChannelFirstD, AddChannelD, ScaleIntensityD, ToTensorD, Compose, CropForegroundd,\
    AsDiscreteD, SpacingD, OrientationD, ResizeD, RandAffineD, CopyItemsd, OneOf, RandCoarseDropoutd, RandFlipd
from monai.data import CacheDataset

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################################################################################################################################# load data

# load data and output folder paths
dossier = sys.argv[1]
outdir = sys.argv[2]

# check wether weights are indicated and load them, if we continue a training
weights = None
if len(sys.argv)>4:
    weights = sys.argv[4]

if not os.path.exists(outdir):
    os.makedirs(outdir)   

KEYS = ("cerveau", "GT")

# Create dataloaders
xform = Compose([LoadImageD(KEYS),
    EnsureChannelFirstD(KEYS),
    ToTensorD(KEYS)])

bs = 12


# load 6 groups of data, made based on the difficulty metric
# path "6" for the easiest data, those data are also in "5", "4", "3", "2" and "1"
# path "1" for the totality of the data

# training

train_dir1 = dossier + '/train/6/CT/'
train_dir_label1 = dossier + '/train/6/GT/'

train_images1 = sorted(glob.glob(train_dir1 + "*.jpg"))
train_labels1 = sorted(glob.glob(train_dir_label1 + "*.png"))

train_files1 = [{"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(train_images1, train_labels1)]




train_dir2 = dossier + '/train/5/CT/'
train_dir_label2 = dossier + '/train/5/GT/'

train_images2 = sorted(glob.glob(train_dir2 + "*.jpg"))
train_labels2 = sorted(glob.glob(train_dir_label2 + "*.png"))

train_files2 = [{"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(train_images2, train_labels2)]



train_dir3 = dossier + '/train/4/CT/'
train_dir_label3 = dossier + '/train/4/GT/'

train_images3 = sorted(glob.glob(train_dir3 + "*.jpg"))
train_labels3 = sorted(glob.glob(train_dir_label3 + "*.png"))

train_files3 = [{"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(train_images3, train_labels3)]



 
train_dir4 = dossier + '/train/3/CT/'
train_dir_label4 = dossier + '/train/3/GT/'

train_images4 = sorted(glob.glob(train_dir4 + "*.jpg"))
train_labels4 = sorted(glob.glob(train_dir_label4 + "*.png"))

train_files4 = [{"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(train_images4, train_labels4)]




train_dir5 = dossier + '/train/2/CT/'
train_dir_label5 = dossier + '/train/2/GT/'

train_images5 = sorted(glob.glob(train_dir5 + "*.jpg"))
train_labels5 = sorted(glob.glob(train_dir_label5 + "*.png"))

train_files5 = [{"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(train_images5, train_labels5)]




train_dir6 = dossier + '/train/1/CT/'
train_dir_label6 = dossier + '/train/1/GT/'

train_images6 = sorted(glob.glob(train_dir6 + "*.jpg"))
train_labels6 = sorted(glob.glob(train_dir_label6 + "*.png"))

train_files6 = [{"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(train_images6, train_labels6)]



# validation

val_dir1 = dossier + '/val/6/CT/'
val_dir_label1 = dossier + '/val/6/GT/'

val_images1 = sorted(glob.glob(val_dir1 + "*.jpg"))
val_labels1 = sorted(glob.glob(val_dir_label1 + "*.png"))

val_files1 = [{"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(val_images1, val_labels1)]



val_dir2 = dossier + '/val/5/CT/'
val_dir_label2 = dossier + '/val/5/GT/'

val_images2 = sorted(glob.glob(val_dir2 + "*.jpg"))
val_labels2 = sorted(glob.glob(val_dir_label2 + "*.png"))

val_files2 = [{"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(val_images2, val_labels2)]



val_dir3 = dossier + '/val/4/CT/'
val_dir_label3 = dossier + '/val/4/GT/'

val_images3 = sorted(glob.glob(val_dir3 + "*.jpg"))
val_labels3 = sorted(glob.glob(val_dir_label3 + "*.png"))

val_files3 = [{"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(val_images3, val_labels3)]



val_dir4 = dossier + '/val/3/CT/'
val_dir_label4 = dossier + '/val/3/GT/'

val_images4 = sorted(glob.glob(val_dir4 + "*.jpg"))
val_labels4 = sorted(glob.glob(val_dir_label4 + "*.png"))

val_files4 = [{"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(val_images4, val_labels4)]



val_dir5 = dossier + '/val/2/CT/'
val_dir_label5 = dossier + '/val/2/GT/'

val_images5 = sorted(glob.glob(val_dir5 + "*.jpg"))
val_labels5 = sorted(glob.glob(val_dir_label5 + "*.png"))

val_files5 = [{"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(val_images5, val_labels5)]



val_dir6 = dossier + '/val/1/CT/'
val_dir_label6 = dossier + '/val/1/GT/'

val_images6 = sorted(glob.glob(val_dir6 + "*.jpg"))
val_labels6 = sorted(glob.glob(val_dir_label6 + "*.png"))

val_files6 = [{"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(val_images6, val_labels6)]




    

################################################################################################################ training parameters
# learning rate
lr = float(sys.argv[3])


# Number of epochs
num_epoch = 200



# ############################################################################################################### training
from generator_UNet import UNet
# Summary of the generator, adapt the size to the images
summary(UNet().cuda(), (1, 192, 192))

from train_generator_CL_val import train_net
generator = train_net(train_files1, train_files2, train_files3, train_files4, train_files5, train_files6, val_files1, val_files2, val_files3, val_files4, val_files5, val_files6, bs, xform, outdir, num_epoch=num_epoch, lr=lr)


