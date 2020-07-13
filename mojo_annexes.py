#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:43:12 2020

@author: vchantreau
Mojo test

Sperm counting from video
annexe file to describe each step
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
import mojo_chantreau


# main function to get the number of spermatozoid from a sequence
mojo_chantreau.process_n_img_from_video('mojo_video1.avi', 
                                        nb_im=10, 
                                        mov_dist=40, 
                                        plot_res=True, 
                                        obj_area=400, 
                                        kept_dist=30)

# steps detailled bellow

# video as a list of images
container = av.open('mojo_video1.avi')
nb_im = 5
count_im = nb_im
images = []
for frame in container.decode(video=0):
    while count_im > 0:
        img = frame.to_image() #
        arr = np.asarray(img) 
        images.append(arr)
        count_im -= 1

# visualization of the raw image and the processed images
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
axs[0,0].imshow(gray, cmap=plt.cm.gray)
axs[0,0].set_title("grayscale")
axs[0,1].imshow(log_corr, cmap=plt.cm.gray)
axs[0,1].set_title("logarithmic correction")
axs[1,0].imshow(bin_im, cmap=plt.cm.gray)
axs[1,0].set_title("binarize with 101 block_size")
axs[1,1].imshow(bin_im1, cmap=plt.cm.gray)
axs[1,1].set_title("binarize with 121 block_size")
axs[0,2].imshow(equilize, cmap=plt.cm.gray)
axs[0,2].set_title("local contrast enhancement")
axs[1,2].imshow(bin_im2, cmap=plt.cm.gray)
axs[1,2].set_title("binarize with 81 block_size")
plt.show()

# get the mean size of one spermatozoïd
bin_im = mojo_chantreau.get_bin(images[0])
# let's approximate the area for one spermatozoïd
# take into account noise from spermatozoïds from the background
plt.figure(0)
plt.imshow(bin_im[900:1100,1300:1500]) # about 3 spermatozoïds
plt.show()
plt.figure(1)
plt.imshow(bin_im[1075:1200,900:1050]) # about 2 spermatozoïds
plt.show()
plt.figure(2)
plt.imshow(bin_im[500:700,875:1100]) # about 4 spermatozoïds
plt.show()
areas = [np.sum(bin_im[500:700,875:1100])/3,
         np.sum(bin_im[1075:1200,900:1050])/2,
         np.sum(bin_im[900:1100,1300:1500])/4]
area = round(np.mean(areas))

# get a representation to clear noise so one image can be compared to another
# this representation is based on the clustering of the object
# with kmeans methods
centers = mojo_chantreau.get_object_center(bin_im)
# visualization of the centers of the clusters
centersT = np.array(centers).transpose()
plt.figure(3)
plt.imshow(images[0])
plt.scatter(centersT[1], centersT[0], color='red', s=20, marker="x")
plt.show()

# visualisation of the clusters (take 1-2 minutes !!!)
nb_objects = mojo_chantreau.get_nb_object(bin_im,400)
all_pos = np.argwhere(bin_im > 0)
all_posT = all_pos.transpose()
# KMeans algorithm to affect each point to one cluster
kmeans_model = KMeans(n_clusters=nb_objects).fit(all_pos)
colors = mcolors.CSS4_COLORS
col_names = list(colors)
plt.figure(2)
for i, l in enumerate(kmeans_model.labels_):
    print(i/nb_objects)
    plt.scatter(all_posT[1][i], all_posT[0][i], color=colors[col_names[l]], s=2)
    plt.xlim([0,bin_im.shape[1]])
    plt.ylim([bin_im.shape[0],0])
plt.show()

