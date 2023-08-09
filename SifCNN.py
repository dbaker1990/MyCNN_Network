import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import time

haveCuda  = torch.cuda.is_available()
print(torch.cuda.get_device_name(0))

batch_size = 32
n_epochs = 20
learning_rate = 0.001
