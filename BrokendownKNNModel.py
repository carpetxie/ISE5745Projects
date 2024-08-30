from google.colab import files
uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(name = fn, length =len(uploaded[fn])))

  from __future__ import division
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
import numpy as np
import scipy as sp
from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""Helper image-processing code."""
def image_to_matrix(image_file, grays=False):
    """
    Convert .png image to matrix
    of values.

    params:
    image_file = str
    grays = Boolean

    returns:
    img = (color) np.ndarray[np.ndarray[np.ndarray[float]]]
    or (grayscale) np.ndarray[np.ndarray[float]]
    """
    img = image.imread(image_file)
    # in case of transparency values
    if(len(img.shape) == 3 and img.shape[2] > 3):
        height, width, depth = img.shape
        new_img = np.zeros([height, width, 3])
        for r in range(height):
            for c in range(width):
                new_img[r,c,:] = img[r,c,0:3]
        img = np.copy(new_img)
    if(grays and len(img.shape) == 3):
        height, width = img.shape[0:2]
        new_img = np.zeros([height, width])
        for r in range(height):
            for c in range(width):
                new_img[r,c] = img[r,c,0]
        img = new_img
    # clean up zeros
    if(len(img.shape) == 2):
        zeros = np.where(img == 0)[0]
        img[zeros] += 1e-7
    return img

def matrix_to_image(image_matrix, image_file):
    """
    Convert matrix of color/grayscale
    values  to .png image
    and save to file.

    params:
    image_matrix = (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] or (grayscale) numpy.ndarray[numpy.ndarray[float]]
    image_file = str
    """
    # provide cmap to grayscale images
    cMap = None
    if(len(image_matrix.shape) < 3):
        cMap = cm.Greys_r
    image.imsave(image_file, image_matrix, cmap=cMap)
    #image.imsave(image_file, image_matrix, cmap=cm.Greys_r)

def image_width(image_matrix):
    if(len(image_matrix.shape) == 3):
        height, width, depth = image_matrix.shape
    else:
        height, width = image_matrix.shape
    return width

def flatten_image_matrix(image_matrix):
    """
    Flatten image matrix from
    Height by Width by Depth
    to (Height*Width) by Depth
    matrix.

    params:
    image_matrix = (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] or (grayscale) numpy.ndarray[numpy.ndarray[float]]

    returns:
    flattened_values = (color) numpy.ndarray[numpy.ndarray[float]] or (grayscale) numpy.ndarray[float]
    """
    if(len(image_matrix.shape) == 3):
        height, width, depth = image_matrix.shape
    else:
        height, width = image_matrix.shape
        depth = 1
    flattened_values = np.zeros([height*width,depth])
    for i, r in enumerate(image_matrix):
        for j, c in enumerate(r):
            flattened_values[i*width+j,:] = c
    return flattened_values

def unflatten_image_matrix(image_matrix, width):
    """
    Unflatten image matrix from
    (Height*Width) by Depth to
    Height by Width by Depth matrix.

    params:
    image_matrix = (color) numpy.ndarray[numpy.ndarray[float]] or (grayscale) numpy.ndarray[float]
    width = int

    returns:
    unflattened_values = (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] or (grayscale) numpy.ndarray[numpy.ndarray[float]]
    """
    heightWidth = image_matrix.shape[0]
    height = int(heightWidth / width)
    if(len(image_matrix.shape) > 1):
        depth = image_matrix.shape[-1]
        unflattened_values = np.zeros([height, width, depth])
        for i in range(height):
            for j in range(width):
                unflattened_values[i,j,:] = image_matrix[i*width+j,:]
    else:
        depth = 1
        unflattened_values = np.zeros([height, width])
        for i in range(height):
            for j in range(width):
                unflattened_values[i,j] = image_matrix[i*width+j]
    return unflattened_values

def image_difference(image_values_1, image_values_2):
    """
    Calculate the total difference
    in values between two images.
    Assumes that both images have same
    shape.

    params:
    image_values_1 = (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] or (grayscale) numpy.ndarray[numpy.ndarray[float]]
    image_values_2 = (color) numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] or (grayscale) numpy.ndarray[numpy.ndarray[float]]

    returns:
    dist = int
    """
    flat_vals_1 = flatten_image_matrix(image_values_1)
    flat_vals_2 = flatten_image_matrix(image_values_2)
    N, depth = flat_vals_1.shape
    dist = 0.
    point_thresh = 0.005
    for i in range(N):
        if(depth > 1):
            new_dist = sum(abs(flat_vals_1[i] - flat_vals_2[i]))
            if(new_dist > depth * point_thresh):
                dist += new_dist
        else:
            new_dist = abs(flat_vals_1[i] - flat_vals_2[i])
            if(new_dist > point_thresh):
                dist += new_dist
    return dist




from random import randint
from math import sqrt
from functools import reduce

import numpy as np

def k_means_cluster(image_values, k=4):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """
    def distance_point_mean(pix,clustMean):
        d = sqrt( (pix[0]-clustMean[0])*(pix[0]-clustMean[0]) + (pix[1]-clustMean[1])*(pix[1]-clustMean[1]) + (pix[2]-clustMean[2])*(pix[2]-clustMean[2]) )
        return d

    # Flatten the image matrix
    imHeight,imWidth,imDep = np.shape(image_values)
    flat_image = flatten_image_matrix(image_values)
    numPix, rgbdim = np.shape(flat_image)
    updated_image_values = np.copy(flat_image)

    # random initiliaziaiton for cluster
    initial_means = []
    np.random.seed(42) #reproducibility
    for i in range(k):
      random_i = np.random.randint(0, flat_image.shape[0])
      initial_means.append(flat_image[random_i])

    # cluster track
    assignments = np.zeros(flat_image.shape[0], dtype=np.int32)
    updated_image_values = np.zeros((flat_image.shape[0], flat_image.shape[1]))

    # k-mean
    for num in range(3):
        for i, pixel in enumerate(flat_image):
          distances = np.array([distance_point_mean(pixel, mean) for mean in initial_means]) #generator functions to minimize run time
          cluster = np.argmin(distances)
          assignments[i] = cluster

        new_means = np.array([flat_image[assignments == j].mean(axis=0) for j in range(k)])

        # convergence
        ''' # sometimes works. use when iterations reaches higher numbers
        if np.all(new_means == initial_means):
            break
        '''
        initial_means = new_means

    # new pic values
    for i, cluster in enumerate(assignments):
        updated_image_values[i] = initial_means[cluster]
    updated_image_values_sqr = unflatten_image_matrix(updated_image_values, image_width(image_values))

    return updated_image_values_sqr

def k_means_test():
    """
    Testing your implementation
    of k-means on the segmented
    reference images.
    """

    # Identify the location of the input image.
    # Load the image in as a matrix.
    #image_dir = 'images/'
    image_name = 'lily.png'
    image_values = image_to_matrix(image_name)
    '''
    if (image_values>1).any():
      image_values = image_values / 255
    ''' #for EC iv b/c my pic isn't flattened to 0 --> 1
    print(image_values.shape)

    # Define how many (k) clusters you want.
    # For debugging, I suggest you use k == 2.
    #k = 2
    k = 4

    # Call k-means.
    updated_values = k_means_cluster(image_values, k)

    # Convert the output pixels from matrix to image. Save the output image.
    ref_image ='k%d_%s'%(k, image_name)
    matrix_to_image(updated_values, ref_image)


k_means_test()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

images = image_to_matrix('lily.png')
flattened = flatten_image_matrix(images)
X = flattened[:,0]
Y = flattened[:,1]
Z = flattened[:,2]

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, s=0.1)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

plt.show()

