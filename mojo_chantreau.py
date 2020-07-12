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
for frame in container.decode(video=0):
    img = frame.to_image() #.save('frame-%04d.jpg' % frame.index)
    arr = np.asarray(img) 
    break

# is there a channel better than the other ?
# Split
red = arr[:, :, 0]
green = arr[:, :, 1]
blue = arr[:, :, 2]
gray = rgb2gray(arr)
# Plot
fig, axs = plt.subplots(2,2)

#cax_00 = axs[0,0].imshow(rgb2gray(arr), cmap=plt.cm.gray)
cax_00 = axs[0,0].imshow(arr)
axs[0,0].xaxis.set_major_formatter(plt.NullFormatter())  # kill xlabels
axs[0,0].yaxis.set_major_formatter(plt.NullFormatter()) 

cax_01 = axs[0,1].imshow(red, cmap='Reds')
fig.colorbar(cax_01, ax=axs[0,1])
axs[0,1].xaxis.set_major_formatter(plt.NullFormatter())
axs[0,1].yaxis.set_major_formatter(plt.NullFormatter())

cax_10 = axs[1,0].imshow(green, cmap='Greens')
fig.colorbar(cax_10, ax=axs[1,0])
axs[1,0].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,0].yaxis.set_major_formatter(plt.NullFormatter())

cax_11 = axs[1,1].imshow(blue, cmap='Blues')
fig.colorbar(cax_11, ax=axs[1,1])
axs[1,1].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,1].yaxis.set_major_formatter(plt.NullFormatter())
plt.show()

# there is a hallo in the center of the image
# try a normalisation
logarithmic_corrected = exposure.adjust_log(gray, 0.8)
equilize = exposure.equalize_adapthist(gray)
# local threshold to get mask of objects
ad_thresh = threshold_local(equilize, 101, offset=0.07) # is the best one
bin_im = equilize > ad_thresh
ad_thresh1 = threshold_local(equilize, 121, offset=0.07) #35
bin_im1 = equilize > ad_thresh1
ad_thresh2 = threshold_local(equilize, 81, offset=0.07) #35
bin_im2 = equilize > ad_thresh2

# Plot
fig, axs = plt.subplots(2,2)

#cax_00 = axs[0,0].imshow(rgb2gray(arr), cmap=plt.cm.gray)
cax_00 = axs[0,0].imshow(gray, cmap=plt.cm.gray)
axs[0,0].xaxis.set_major_formatter(plt.NullFormatter())  # kill xlabels
axs[0,0].yaxis.set_major_formatter(plt.NullFormatter()) 

cax_01 = axs[0,1].imshow(bin_im, cmap=plt.cm.gray)
#axs[0,1].xaxis.set_major_formatter(plt.NullFormatter())
#axs[0,1].yaxis.set_major_formatter(plt.NullFormatter())

cax_10 = axs[1,0].imshow(bin_im1, cmap=plt.cm.gray)
axs[1,0].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,0].yaxis.set_major_formatter(plt.NullFormatter())

cax_00 = axs[1,1].imshow(bin_im2, cmap=plt.cm.gray)
axs[1,1].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,1].yaxis.set_major_formatter(plt.NullFormatter())
plt.show()

# let's approximate an area for one spermatozoïde
plt.imshow(bin_im[900:1100,1300:1500]) #2
plt.imshow(bin_im[1075:1200,900:1050]) #1
plt.imshow(bin_im[500:700,875:1100]) #2
plt.show()

areas = [bin_im[500:700,875:1100].sum().sum()/2,
         bin_im[1075:1200,900:1050].sum().sum(),
         bin_im[900:1100,1300:1500].sum().sum()/2]

area = round(np.mean(areas))