# KMeans Clustering for Schema Map overlay of original images dataset. For use/preprocessing BEFORE the CNN.
# Algorithm finds rare colors in a picture and draws lines between them to make a 'map' of features where data might
# otherwise not be interpretable as being a feature or have a relationship via a CNN.

import tensorflow as tf
import numpy as np
#import os
#import math
#import timeit
import matplotlib.pyplot as plt
import cv2 as cv2

def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):

    # Load the raw CIFAR-10 dataset
    cifar10 = tf.keras.datasets.cifar10.load_data()

    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    # Go through training and testing images/datasets
    for dataset in range(2):
        if dataset == 0:
            dataset_size = (np.size(X_train, 0))
            dset = X_train
        elif dataset == 1:
            dataset_size = (np.size(X_test, 0))
            dset = X_test

        modified_set_of_images = np.zeros(dset.shape)
        for i in range(dataset_size): # should be dataset_size for range

            # KMeans of image to find rare colors
            image = dset[i]
            flattened_image = image.reshape(-1, 3)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            flags = cv2.KMEANS_RANDOM_CENTERS
            K = 150        # PARAMETER: Number of centroids/clusters
            attempts = 10  # PARAMETER: number of attempts to min loss amongst centroids via random centeroid placement
            compactness,labels,centers = cv2.kmeans(flattened_image,K,None,criteria,attempts,flags)
            image_labels = labels.reshape([np.shape(image)[0],np.shape(image)[1]])
            rare_pixel_locations_list = []
            for j in range(np.shape(image)[0]):
                    for k in range(np.shape(image)[1]):
                        if image_labels[j][k] < 1:    # PARAMETER: labels/centeroids with few datapoints means rare color
                            rare_pixel_locations_list.append([j, k])

            # draw lines between all rare color pixels
            modified_img = image
            for pixel1 in range(len(rare_pixel_locations_list)):
                for pixel2 in range(len(rare_pixel_locations_list)):
                    modified_img = cv2.line(modified_img, tuple(rare_pixel_locations_list[pixel1]), tuple(rare_pixel_locations_list[pixel2]), (50,200,50), 1)

            # overwrite the image data set with the modified picture data sets
            modified_img = modified_img.get() # must use get() to pull out of GPU and into CPU
            modified_img = np.asarray(modified_img, dtype=np.float32)
            modified_set_of_images[i] = modified_img
            if dataset == 0:
                X_train = modified_set_of_images
            if dataset == 1:
                X_test = modified_set_of_images

    #plt.show(plt.imshow(X_train[48999].astype('uint8')))
    #plt.show(plt.imshow(X_test[9999].astype('uint8')))
