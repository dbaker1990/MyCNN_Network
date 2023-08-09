import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import time

#Check to see if cuda is available if so show the name of the gpu
haveCuda  = torch.cuda.is_available()
print(torch.cuda.get_device_name(0))

batch_size = 32
n_epochs = 10
learning_rate = 0.001
shuffleImages = True

#Set the training folder and set labels for each folder
data = ImageFolder(root="./train")
labels = []
for x in data.classes:
    labels.append(x)
print(labels)
