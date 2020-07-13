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
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.colors as mcolors

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
bin_im = equilize < ad_thresh
ad_thresh1 = threshold_local(equilize, 121, offset=0.07) #35
bin_im1 = equilize < ad_thresh1
ad_thresh2 = threshold_local(equilize, 81, offset=0.07) #35
bin_im2 = equilize < ad_thresh2

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
    bin_im = equilize < ad_thresh
    return bin_im

bin_im = get_bin(images[0])
# let's approximate the area for one spermatozoïd
# take into account noise from spermatozoïds from the background
plt.imshow(bin_im[900:1100,1300:1500]) # about 3 spermatozoïds
plt.imshow(bin_im[1075:1200,900:1050]) # about 2 spermatozoïds
plt.imshow(bin_im[500:700,875:1100]) # about 4 spermatozoïds
plt.show()

areas = [np.sum(bin_im[500:700,875:1100])/3,
         np.sum(bin_im[1075:1200,900:1050])/2,
         np.sum(bin_im[900:1100,1300:1500])/4]

area = round(np.mean(areas))

def get_nb_object(img, obj_area):
    return int(round(np.sum(img)/obj_area))

nb_objects = get_nb_object(bin_im, area)

# get a representation to clear noise so one image can be compared to another
all_pos = np.argwhere(bin_im > 0)
all_posT = all_pos.transpose()
plt.figure(0)
plt.scatter(all_posT[1], all_posT[0],s=2)
plt.ylim([bin_im.shape[0],0])
plt.show()
# for comparison
plt.figure(1)
#plt.imshow(bin_im, cmap=plt.cm.gray)
plt.imshow(images[0])
plt.show()

# KMeans algorithm to affect each point to one cluster => too long
kmeans_model = KMeans(n_clusters=nb_objects).fit(all_pos)
colors = mcolors.CSS4_COLORS
col_names = list(colors)
plt.figure(2)
for i, l in enumerate(kmeans_model.labels_):
    print(i)
    plt.scatter(all_posT[1][i], all_posT[0][i], color=colors[col_names[l]], s=2)#, marker=markers[l],ls='None') # color=colors[l]
    plt.xlim([0,bin_im.shape[1]])
    plt.ylim([bin_im.shape[0],0])
plt.show()

# get the cluster centroïd to get a representation
centers = np.array(kmeans_model.cluster_centers_)
centersT = centers.transpose()
plt.figure(3)
#plt.imshow(bin_im, cmap=plt.cm.gray)
plt.imshow(images[0])
plt.scatter(centersT[1], centersT[0], color='red', s=20, marker="x")
plt.show()

def get_object_center(bin_im, area = 490):
    nb_objects = get_nb_object(bin_im, area)
    all_pos = np.argwhere(bin_im > 0)
    kmeans_model = KMeans(n_clusters=nb_objects).fit(all_pos)
    return np.array(kmeans_model.cluster_centers_)

def plot_centers(img, centers):
    centersT = centers.transpose()
    plt.figure(0)
    plt.imshow(img)
    plt.scatter(centersT[1], centersT[0], color='red', s=20, marker="x")
    plt.show()
    
    


# density plot
# Extract x and y
x = all_pos[:, 0]
y = all_pos[:, 1]# Define the borders
deltaX = (max(x) - min(x))/10
deltaY = (max(y) - min(y))/10xmin = min(x) - deltaX
xmax = max(x) + deltaXymin = min(y) - deltaY
ymax = max(y) + deltaYprint(xmin, xmax, ymin, ymax)# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

# fit gaussian kernel
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

# ploting kernel with contours
fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D Gaussian Kernel density estimation')