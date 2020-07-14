# Mojo

Mojo test: count spermatozoïds from video

* mojo_chantreau: main file with documented functions
* mojo_annexes: usage example and illustration of each step of the process of counting

## Prerequisite

* Python 3.8

Librairies:

* numpy==1.18.1
* scikit-image==0.17.2
* av==8.0.2
* scikit-learn==0.23.1

## Quick test

```
import mojo_chantreau
mojo_chantreau.process_n_img_from_video(video_file='mojo_video1.avi', nb_im=10, mov_dist=40, plot_res=True, obj_area=400, kept_dist=30)
```

With **video_file** the path to the video file, **nb_im** the number of images to be extracted and processed from the video (default 10), **mov_dist** the max distance in pixel within wich an object can be moving between two successive images (default 40), **plot_res** to plot the finding object on the last image, **obj_area** the size in pixels of one object (default 400), **kept_dist** the distance within final founding objects will be considered the same unique object (and count as one) (default 30).

## Method

Superpervised learning is not an option, so we need to use statistical methods and unsupervised learning

1. Get nb_im images from video (default: 10)

2. get a number of objects for one image

2.2 enhance contrast values for the image

2.3 binarize the image

2.4 Thank to a mean area obj_area for one spermatozoïd (default: 400), and the total area for the image, deduce the number of objects

3. get a representation of the object of this image to get rid of the noise, so it can be compare to one another.
All points are map to n clusters, then the centroïd of each cluster are taken for representing the n objects.

3.1 Nb of object = nb of clusters

3.2 kmeans to affect each black pixel to one cluster

3.3 get the centroïds (possible biais due to superimposed objects)

4. Compare each image representation to the next one, to assess static objects, moving objects and appearing/disappearing objects (news/losts) (due to focal or border)

5. Merge the lists (losts, statics, movings, news) obtained while processing the nb_im and mean the closest points (within kept_dist distance) to get rid of redundancies

6. The lenght of the resulting list is the number of spermatozoïds for this video !

## Conclusion and further improvements

- Some parameters needs to be improved, for example try to reduce the obj_area size.
- the kmeans clustering works well, but we can try with density instead, the pics (gaussian kernels) being an object, to check on precision and performance.
- the image processing need to be enhance, by removing or cropping the artefact like in video 2 (try edge detection and cropping the outside).


