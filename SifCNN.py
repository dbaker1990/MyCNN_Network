import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import time

haveCuda  = torch.cuda.is_available()
print(torch.cuda.get_device_name(0))