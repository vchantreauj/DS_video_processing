# Mojo
Mojo test: count spermatozoïds from video

Superpervised learning is not an option, so we need to use statistical methods and unsupervised learning

1. Work on one image to get a number of objects

1.1 get images from video

1.2 equilize values for one image

1.3 find best parameters to binarize it so the background is gone

1.4 get the mean area of one object

1.5 With the total area, deduce the number of object

2. Normalize this image in a simple representation to get rid of the noise, so it can be compare to one another

2.1 Nb of object = nb of clusters

2.2 kmeans to affecteach black pixel to one cluster

2.3 Calcul the centroïds

2.4 The matrix with the centroïds is a representation of the image, with each centroïd supposed to be one object

3. Compare each image representation to the next one, to assess static object, moving object and appearing/disappearing object (due to focal or border)


