import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2 
import matplotlib.pyplot as plt
import time
import ImageResize

#Check to see if cuda is available if so show the name of the gpu
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))

batch_size = 32
n_epochs = 10
learning_rate = 0.0015
shuffleImages = True
num_classes = 4


#transforms
itransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
#Set the training folder and set labels for each folder
trainset = ImageFolder(root='./rgbtrain',transform=itransform)
labels = []
for x in trainset.classes:
    labels.append(x)

#this is the validation training folder
validset = ImageFolder(root="./test",transform=itransform)


#training and validation data loaders
train_loader = DataLoader(trainset,batch_size,shuffleImages, num_workers=2)
valid_loader = DataLoader(validset,batch_size,shuffleImages, num_workers=2)

#This is function I created it resize images to the nearest cubic
#ImageResize.ResizeImage(640,480,"imagePath")

class Sif(nn.Module):
    def __init__(self, num_classes):
        super(Sif, self).__init__()
        
        #convolutional layers
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv_layer3 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ce1 = nn.CELU()
        
        self.conv_layer5 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer6 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv_layer7 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer8 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ce2 = nn.CELU()
        
        self.conv_layer9 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv_layer10 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
        self.conv_layer11 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv_layer12 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ce3 = nn.CELU()
        
        self.conv_layer13 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv_layer14 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv_layer15 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer16 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv_layer17 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ce4 = nn.CELU()
        
        self.fc1 = nn.Linear(1600,128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool1(out)
        out = self.ce1(out)
        
        out = self.conv_layer5(out)
        out = self.conv_layer6(out)
        out = self.conv_layer7(out)
        out = self.conv_layer8(out)
        out = self.max_pool2(out)
        out = self.ce2(out)
        
        out = self.conv_layer9(out)
        out = self.conv_layer10(out)
        out = self.conv_layer11(out)
        out = self.conv_layer12(out)
        out = self.max_pool3(out)
        out = self.ce3(out)
        
        out = self.conv_layer13(out)
        out = self.conv_layer14(out)
        out = self.conv_layer15(out)
        out = self.conv_layer16(out)
        out = self.conv_layer17(out)
        out = self.max_pool4(out)
        out = self.ce4(out)
        
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        
        return out
    

model = Sif(num_classes)
model.cuda()

#set loss function with criterion
criterion = nn.CrossEntropyLoss()

#Set optimizer with optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

total_step = len(train_loader)

#Use the predefined number of epochs to determine how many iterations to train the network on
for epoch in range(n_epochs):
    #load in the data in batches using the train_loader object
    for i, (images,labels) in enumerate(train_loader):
        # move tensors to the configured device
        images, labels = images.to(device), labels.to(device)
        
        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, loss.item()))
        
        
