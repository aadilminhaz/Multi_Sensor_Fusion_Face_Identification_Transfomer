import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision.utils import make_grid

import os
import pandas as pd
import torch
from skimage import io
from PIL import Image
import torch.utils.data

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

import Utils

# Path to the directory containing images
'''image_dir = 'data/IR_RGB_ds'
rgb_image_dir = 'data/IR_RGB_ds/RGB-faces-128x128/'
ir_image_dir = 'data/IR_RGB_ds/thermal-face-128x128/'

#Dataset parameters#
batch_size = 32
mapping_file = 'ir_rgb_label.csv'''

class IR_RGB_Dataset(Dataset):
    def __init__(self, rgb_source, ir_source, data_source_file = None, rgb_transform=None, ir_transform=None):
        self.annotations = pd.read_csv(data_source_file)
        self.root_dir_rgb = rgb_source
        self.root_dir_ir = ir_source
        self.rgb_transform = rgb_transform
        self.ir_transform = ir_transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        rgb_img_path = os.path.join(self.root_dir_rgb, self.annotations.iloc[index, 0])
        #print('Loading rbg_image : ', rgb_img_path)
        rgb_image = Image.open(rgb_img_path)
        ##print('rbg_image loaded  ', rgb_image.shape)
        ir_img_path = os.path.join(self.root_dir_ir, self.annotations.iloc[index, 1])
        #print('Loading ir_image : ', ir_img_path)
        ir_image = Image.open(ir_img_path)
        #print('ir_image loaded  ', ir_image.shape)
        y_label = torch.tensor(int(self.annotations.iloc[index, 2]))
        #print('label loaded   ', y_label)

        if self.rgb_transform is not None:
            #print('Appying rgb_tranfom')
            rgb_image = self.rgb_transform(rgb_image)
            #print('Apllied rgb_transform :', rgb_image.shape)
        if self.ir_transform is not None:
            #print('Appying ir_tranfom')
            ir_image = self.ir_transform(ir_image)
            #print('Applied ir_tranfom : ', ir_image.shape)
        if not isinstance(rgb_image, torch.Tensor):
            rgb_image = transforms.ToTensor()(rgb_image)
        if not isinstance(ir_image, torch.Tensor):
            ir_image = transforms.ToTensor()(ir_image)
        return rgb_image, ir_image, y_label    

def load_IR_RGB_dataset(rgb_source, ir_source, data_source_file, batch_size=32, rgb_transform=None, ir_transform=None):

    dataset = IR_RGB_Dataset(rgb_source, ir_source, data_source_file, rgb_transform, ir_transform)

    gen = torch.Generator()
    gen.manual_seed(42)
    #train_set, test_set = torch.utils.data.random_split(
    #    dataset, [1350, 182],
    #    generator=gen   
    #    )
    
    train_set = torch.utils.data.Subset(dataset, range(1350))
    test_set = torch.utils.data.Subset(dataset, range(1350, 1523))

    

    #train_set, test_set = torch.utils.data.random_split(dataset, [900, 603])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader 

def load_IR_RGB_dataset_kfold(rgb_source, ir_source, data_source_file, k_folds=5, batch_size=32, rgb_transform=None, ir_transform=None):

    dataset = IR_RGB_Dataset(rgb_source, ir_source, data_source_file, rgb_transform, ir_transform)

    print('batch_size  :', batch_size)
    #kf = KFold(n_splits=k_folds, shuffle=False, random_state=42)
    kf = KFold(n_splits=k_folds, shuffle=False)
    
    train_loaders = []
    val_loaders = []

    for train_idx, val_idx in kf.split(dataset):
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders


'''def load_new_IR_RGB_dataset(data_source_file=None, rgb_image_dir=None, ir_image_dir=None):
    print('new train file :', data_source_file)
    image_dir = 'data/IR_RGB_ds'
    if rgb_image_dir is None:
        rgb_image_dir = 'data/IR_RGB_ds/eval_new_data_train/RGB/'
    if ir_image_dir is None:
        ir_image_dir = 'data/IR_RGB_ds/eval_new_data_train/IR/'

    rgb_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #transforms.Normalize((0.1307,), (0.3081,))
    ])

    ir_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    
        #transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = IR_RGB_Dataset(csv_file=new_csv_file, root_dir_rgb=rgb_image_dir, root_dir_ir=ir_image_dir,  rgb_transform=rgb_transform, ir_transform=ir_transform)

    #train_set, test_set = torch.utils.data.random_split(dataset, [900, 603])
    train_set = dataset
    train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=True)
    #test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    return train_loader 
'''

'''
def pair_show_images(rgb_images, ir_images, labels, image_at_once=batch_size):
    fig = plt.figure(figsize=(12, 10))  # Adjust the figure size as needed
    for i in range(image_at_once):  # Display 32 images
        plt.subplot(4, 8, i + 1)  # Create a 4x8 grid of subplots#
        plt.imshow(rgb_images[i][0], cmap='gray', interpolation='none')
        #plt.imshow(ir_images[i][0], cmap='gray', interpolation='none')
        plt.title("Name: {}".format(labels[i]))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

'''