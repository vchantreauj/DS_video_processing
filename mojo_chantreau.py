#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 19:31:14 2020

@author: vchantreau
Mojo test

Sperm counting from video
main file with all required function
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

    # get_bin increase contrast and binarize the input image
def get_bin(img):
    gray = rgb2gray(img)
    equilize = exposure.equalize_adapthist(gray)
    # local threshold to get mask of objects
    # considering the size of the image, a quit big block_size works better
    ad_thresh = threshold_local(equilize, block_size = 101, offset=0.07)
    bin_im = equilize < ad_thresh
    return bin_im

    # get_nb_object return the number of spermatozoid in the image imp
    # according to the given size of one spermatozoid obj_area
def get_nb_object(img, obj_area):
    return int(round(np.sum(img)/obj_area))

    # get_object_center perform kmeans clustering on all points from the mask
    # that is to say points belonging to spermatozoid
    # return the center of each cluster (to represent each spermatozoid)
def get_object_center(bin_im, area = 400): #490
    nb_objects = get_nb_object(bin_im, area)
    all_pos = np.argwhere(bin_im > 0)
    kmeans_model = KMeans(n_clusters=nb_objects).fit(all_pos)
    centers = np.round(np.array(kmeans_model.cluster_centers_),0)
    #print(str(len(centers))," spermatozoïds for this image")
    return [list(item) for item in centers] # centers

    # plot_centers plots the center of each cluster on the raw image
def plot_centers(img, centers):
    centersT = centers.transpose()
    plt.figure(0)
    plt.imshow(img)
    plt.scatter(centersT[1], centersT[0], color='red', s=20, marker="x")
    plt.show()
    
    # process_n_img_from_video process nb_im from the given video video_file
    # 1. get the nb_im
    # 2. contrast and binarize images
    # 3. get the representation of the images: one center is one spermatozoid
    # 4. for each image, compare the centers with the previous one so the the center list are split into:
    #       - statics: commons center
    #       - movings: centers which have move around mov_dist from the previous center
    #       - losts: previous center not present in the current image
    #       - news: news center not present in the previous image
    # 5. cluster all points to remove redundancies of points within a distance of kept_dist of one another
    # 6. Plot all points if plot_res = True
def process_n_img_from_video(video_file, nb_im=10, mov_dist=40, plot_res=True, obj_area=400, kept_dist=30):
    container = av.open(video_file)
    # video as a list of images
    count_im = nb_im
    images = []
    for frame in container.decode(video=0):
        while count_im > 0:
            img = frame.to_image() #.save('frame-%04d.jpg' % frame.index)
            arr = np.asarray(img) 
            images.append(arr)
            count_im -= 1
    # compare one image with the following one
    bin_im = get_bin(images[0])
    prev_centers = get_object_center(bin_im, area = obj_area)  
    statics = prev_centers
    movings = []
    losts = []
    for i in range(1, nb_im):
        bin_im = get_bin(images[i])
        centers = get_object_center(bin_im, area = obj_area) 
        # get the static objects
        statics = np.array([x for x in centers if x in statics])
        prev_diff = np.array([x for x in prev_centers if not( x in statics)])
        cur_diff = np.array([x for x in centers if not( x in statics)])
        # get moving object
        # object prev and curr with are in 40 pixels from each other are considered
        # to be the same object moving
        cur_diffT = cur_diff.transpose()
        if i == 1 :
            movings = prev_diff
        movings_prev = []
        movings_cur = []
        for x in movings:
            next_pos = np.where((cur_diffT[0] > (x[0]-mov_dist)) & (cur_diffT[0] < (x[0]+mov_dist)) 
                                & (cur_diffT[1] > (x[1]-mov_dist)) & (cur_diffT[1] < (x[1]+mov_dist)))
            if len(next_pos[0])>0:
                movings_prev.append(x)
                movings_cur.append(cur_diff[next_pos[0][0]])
                # TODO: pick the closest if several return
        movings_prev = np.array(movings_prev)
        movings = np.array(movings_cur)
        # get appearing and disappearing object
        tmp_losts = np.array([x for x in prev_diff if not ((x in movings_prev) or (x in losts))])
        if len(losts) == 0:
            losts = tmp_losts
        else:
            losts = np.append(losts,tmp_losts,0)
        # considere new an object which has never appear before
        news = np.array([x for x in cur_diff if not ((x in movings) or (x in losts))])        
        prev_centers = centers
        
    # remove object too close to one another, because the cluster center may change
    # for the same object during the iteration    
    obj_kept = []
    mov_curT = movings.transpose()
    all_list = np.append(np.append(np.append(losts,movings,0), statics, 0), news, 0) #losts
    while len(all_list)>0:
        # clusters close objects to keep only one per cluster
        x = all_list[0]
        lostsT = np.array(all_list).transpose()
        obj = np.where((lostsT[0] > (x[0]-kept_dist)) & (lostsT[0] < (x[0]+kept_dist)) 
                       & (lostsT[1] > (x[1]-kept_dist)) & (lostsT[1] < (x[1]+kept_dist)))
        if len(obj[0])>1:
            obj_kept.append(list(np.mean(all_list[obj[0]], axis=0)))
        else:
            obj_kept.append(list(x))
        all_list = np.delete(all_list, obj[0], 0)

    if plot_res:
        staticsT = statics.transpose()
        prev_diffT = prev_diff.transpose()
        cur_diffT = cur_diff.transpose()
        mov_prevT = movings_prev.transpose()
        mov_curT = movings.transpose()
        lostsT = np.array(losts).transpose()
        obj_keptT = np.array(obj_kept).transpose()
        newsT = news.transpose() 
        plt.figure(0)
        plt.imshow(images[i])
        plt.scatter(staticsT[1], staticsT[0],color='r',s=15, marker='x')
        plt.scatter(mov_prevT[1], mov_prevT[0],color='b',s=15, marker='o')
        plt.scatter(mov_curT[1], mov_curT[0],color='b',s=15, marker='x')
        plt.scatter(lostsT[1], lostsT[0],color='black',s=15, marker='o')
        plt.scatter(newsT[1], newsT[0],color='g',s=15, marker='x')
        plt.scatter(obj_keptT[1], obj_keptT[0],color='yellow',s=25, marker='+')
        
        plt.ylim([bin_im.shape[0],0])
        plt.legend(labels=['statics','moving last pos', 'moving cur pos','losts','news','kept'])
        plt.show()
    nb_objects = len(obj_kept)
    print('the',nb_im,'images of the video',video_file,'show',str(nb_objects),'spermatozïds.')
    return nb_objects

# process_n_img_from_video('mojo_video1.avi',  nb_im=10, plot_res=True)