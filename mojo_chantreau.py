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
count_im = nb_im
images = []
for frame in container.decode(video=0):
    while count_im > 0:
        img = frame.to_image() #.save('frame-%04d.jpg' % frame.index)
        arr = np.asarray(img) 
        images.append(arr)
        count_im -= 1

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
plt.imshow(images[0])
plt.scatter(centersT[1], centersT[0], color='red', s=20, marker="x")
plt.show()

def get_object_center(bin_im, area = 400): #490
    nb_objects = get_nb_object(bin_im, area)
    all_pos = np.argwhere(bin_im > 0)
    kmeans_model = KMeans(n_clusters=nb_objects).fit(all_pos)
    centers = np.round(np.array(kmeans_model.cluster_centers_),0)
    print(str(len(centers))," spermatozoïds for this image")
    return centers

def plot_centers(img, centers):
    centersT = centers.transpose()
    plt.figure(0)
    plt.imshow(img)
    plt.scatter(centersT[1], centersT[0], color='red', s=20, marker="x")
    plt.show()
    
def get_distance(obj1, obj2):
    return math.sqrt( ((obj1[0]-obj2[0])**2)+((obj1[1]-obj2[1])**2))    
    
centers = get_object_center(bin_im)    
#plot_centers(images[0], centers)    
  
# now we can compare one image with the following one
prev_centers = centers
statics = prev_centers
movings = []
losts = []
for i in range(1, nb_im):
    bin_im = get_bin(images[i])
    centers = get_object_center(bin_im) 
    # get the static objects, to count them only once
    statics = np.array([x for x in centers if x in statics])
    prev_diff = np.array([x for x in prev_centers if not( x in statics)])
    cur_diff = np.array([x for x in centers if not( x in statics)])
    # get moving object, to count them only once
    # object prev and curr with are in 40 pixels from each other are considered
    # to be the same moving object
    # TODO this distance has to be tested, it may be larger
    cur_diffT = cur_diff.transpose()
    if i == 1 :
        movings = prev_diff
    movings_prev = []
    movings_cur = []
    for x in movings:
        next_pos = np.where((cur_diffT[0] > (x[0]-40)) & (cur_diffT[0] < (x[0]+40)) 
                            & (cur_diffT[1] > (x[1]-40)) & (cur_diffT[1] < (x[1]+40)))
        if len(next_pos[0]>0):
            movings_prev.append(x)
            movings_cur.append(cur_diff[next_pos[0][0]])
            # TODO: pick the closest if several return
    movings_prev = np.array(movings_prev)
    movings = np.array(movings_cur)
    # get appearing and disappearing object, to add them all  
    tmp_losts = np.array([x for x in prev_diff if not ((x in movings_prev) or (x in losts))])
    if len(losts) == 0:
        losts = tmp_losts
    else:
        losts = np.append(losts,tmp_losts,0)
    # considere new an object which has never appear before
    news = np.array([x for x in cur_diff if not ((x in movings) or (x in losts))])
    
    prev_centers = centers


staticsT = statics.transpose()
prev_diffT = prev_diff.transpose()
cur_diffT = cur_diff.transpose()
mov_prevT = movings_prev.transpose()
mov_curT = movings.transpose()
lostsT = np.array(losts).transpose()
newsT = news.transpose()

plt.figure(11)
plt.imshow(images[i])
plt.scatter(staticsT[1], staticsT[0],color='r',s=15, marker='x')
plt.scatter(mov_prevT[1], mov_prevT[0],color='b',s=15, marker='o')
plt.scatter(mov_curT[1], mov_curT[0],color='b',s=15, marker='x')
plt.scatter(lostsT[1], lostsT[0],color='black',s=15, marker='o')
plt.scatter(newsT[1], newsT[0],color='g',s=15, marker='x')
plt.ylim([bin_im.shape[0],0])
plt.legend(labels=['statics','moving last pos', 'moving cur pos','losts','news'])
plt.show()

plt.figure(10)
plt.imshow(images[0])
plt.show()



staticsT = statics.transpose()
prev_diffT = prev_diff.transpose()
cur_diffT = cur_diff.transpose()
plt.figure(12)
plt.scatter(staticsT[1], staticsT[0],color='r',s=4, marker='x')
plt.scatter(prev_diffT[1], prev_diffT[0],color='b',s=4, marker='x')
plt.scatter(cur_diffT[1], cur_diffT[0],color='g',s=4, marker='x')
plt.ylim([bin_im.shape[0],0])
plt.show()

'''
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
'''