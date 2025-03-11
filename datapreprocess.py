import os
import torch
from torch.utils import data
from PIL import Image
import torch.nn as nn
import torchvision
import numpy as np
import operator
from medmnist import *
import csv
import cv2
import torchvision.transforms as transforms
# Med MNIST

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=1):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if(self.n_views==1):
            return self.base_transform(x.convert('RGB'))
        elif(self.n_views==2):
            return [self.base_transform(x.convert('RGB')) for i in range(self.n_views)]

class ContrastiveLearningDataset:
    def __init__(self, transform, n_views):
        self.transforms = transform
        self.n_views = n_views

    def get_dataset(self, name):
        # OCTMNIST 
        dataset_t = PathMNIST(split=name, transform=ContrastiveLearningViewGenerator(self.transforms,self.n_views),download=True)
        return dataset_t

