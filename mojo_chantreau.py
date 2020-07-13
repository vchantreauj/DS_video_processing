#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 19:31:14 2020

@author: vchantreau
Mojo test

Sperm counting from video
"""
import numpy as np
import pandas as pd
from PIL import Image
from skimage import data, io, filters
import matplotlib.pyplot as plt
import av
# to install PyAV:
# pip install av
from skimage.color import rgb2gray
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_local

container = av.open('mojo_video1.avi')

# video as a list of images
nb_im = 10
images = []
for frame in container.decode(video=0):
    while nb_im > 0:
        img = frame.to_image() #.save('frame-%04d.jpg' % frame.index)
        arr = np.asarray(img) 
        images.append(arr)
        nb_im -= 1

# visualization
gray = rgb2gray(arr)
# there is a hallo in the center of the image
# try a normalisation
log_corr = exposure.adjust_log(gray, 0.8)
equilize = exposure.equalize_adapthist(gray) # more contrasted than log
# local threshold to get mask of objects
ad_thresh = threshold_local(equilize, 101, offset=0.07) # is the best one
bin_im = equilize > ad_thresh
ad_thresh1 = threshold_local(equilize, 121, offset=0.07) #35
bin_im1 = equilize > ad_thresh1
ad_thresh2 = threshold_local(equilize, 81, offset=0.07) #35
bin_im2 = equilize > ad_thresh2

# Plot
fig, axs = plt.subplots(2,3)
cax_00 = axs[0,0].imshow(gray, cmap=plt.cm.gray)
cax_01 = axs[0,1].imshow(log_corr, cmap=plt.cm.gray)
cax_10 = axs[1,0].imshow(bin_im, cmap=plt.cm.gray)
cax_11 = axs[1,1].imshow(bin_im1, cmap=plt.cm.gray)
cax_02 = axs[0,2].imshow(equilize, cmap=plt.cm.gray)
cax_12 = axs[1,2].imshow(bin_im2, cmap=plt.cm.gray)
plt.show()

# get_bin
# increase contrast and binarize the input image
def get_bin(img):
    gray = rgb2gray(img)
    equilize = exposure.equalize_adapthist(gray)
    # local threshold to get mask of objects
    # considering the size of the image, a quit big block_size works better
    ad_thresh = threshold_local(equilize, block_size = 101, offset=0.07)
    bin_im = equilize > ad_thresh
    return bin_im

bin_im = get_bin(images[0])
# let's approximate the area for one spermatozoïd
plt.imshow(bin_im[900:1100,1300:1500]) #2
plt.imshow(bin_im[1075:1200,900:1050]) #1
plt.imshow(bin_im[500:700,875:1100]) #2
plt.show()

areas = [np.sum(bin_im[500:700,875:1100])/2,
         np.sum(bin_im[1075:1200,900:1050]),
         np.sum(bin_im[900:1100,1300:1500])/2]

area = round(np.mean(areas))