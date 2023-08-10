import os
import cv2
from pathlib import Path
from numpy import asarray
from PIL import Image
import numpy as np

folder_dir = Path('train/Mild Impairment/')
for images in os.listdir(folder_dir):
    wd = os.getcwd()
    if(images.endswith(".jpg")):
        imG = Image.open('train/Mild Impairment/'+images)
        imG = asarray(imG)
        imRGB = np.zeros((imG.shape[0], imG.shape[1], 3), dtype=np.uint8)
        imRGB[:,:,0] = imG
        imRGB[:,:,1] = imG
        imRGB[:,:,2] = imG
        file = images
        os.chdir(Path('rgbtrain/Mild Impairment/'))
        cv2.imwrite(file, imRGB)
        os.chdir(Path(wd))
        
folder_dir = Path('train/Moderate Impairment/')
for images in os.listdir(folder_dir):
    wd = os.getcwd()
    if(images.endswith(".jpg")):
        imG = Image.open('train/Moderate Impairment/'+images)
        imG = asarray(imG)
        imRGB = np.zeros((imG.shape[0], imG.shape[1], 3), dtype=np.uint8)
        imRGB[:,:,0] = imG
        imRGB[:,:,1] = imG
        imRGB[:,:,2] = imG
        file = images
        os.chdir(Path('rgbtrain/Moderate Impairment/'))
        cv2.imwrite(file, imRGB)
        os.chdir(Path(wd))

folder_dir = Path('train/No Impairment/')
for images in os.listdir(folder_dir):
    wd = os.getcwd()
    if(images.endswith(".jpg")):
        imG = Image.open('train/No Impairment/'+images)
        imG = asarray(imG)
        imRGB = np.zeros((imG.shape[0], imG.shape[1], 3), dtype=np.uint8)
        imRGB[:,:,0] = imG
        imRGB[:,:,1] = imG
        imRGB[:,:,2] = imG
        file = images
        os.chdir(Path('rgbtrain/No Impairment/'))
        cv2.imwrite(file, imRGB)
        os.chdir(Path(wd))
        
folder_dir = Path('train/Very Mild Impairment/')
for images in os.listdir(folder_dir):
    wd = os.getcwd()
    if(images.endswith(".jpg")):
        imG = Image.open('train/Very Mild Impairment/'+images)
        imG = asarray(imG)
        imRGB = np.zeros((imG.shape[0], imG.shape[1], 3), dtype=np.uint8)
        imRGB[:,:,0] = imG
        imRGB[:,:,1] = imG
        imRGB[:,:,2] = imG
        file = images
        os.chdir(Path('rgbtrain/Very Mild Impairment/'))
        cv2.imwrite(file, imRGB)
        os.chdir(Path(wd))