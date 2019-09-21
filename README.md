# KMeans-Schema-Map
Adding schema maps to images with openCV before CNN processing via KMeans Clustering of rare colors.

To solve the 'Big Dipper' problem with CNN's I first overlay onto the training and test images lines/maps. These then add meaning and context to aid the CNN in classification.

Big Dipper: A picture of the night sky pointed at the big dipper and then processed by a CNN will generally only result in the basic object detection of 'stars'. In order to detect a big dipper, it would need a schema to literally connect the dots to then provide features that a CNN could then identify.

This is a very basic schema map which uses KMeans to identify rare colors in the images and then use those as special features to connect together with lines. It is a proof of concept, though on the CIFAR10 dataset that I used it performed poorly for the obvious reasons of:

1) Rare colors may be equally as present in the background as on the objects themselves. Therefore, this approach is only good for images with blending issues or where there are interruptions/foreground artifacts etc.
2) The images are too small with dimesions of (32, 32, 3) and therefore the drawing of lines overwhelms the images.

To Run: Copy/Paste this code to the beginning of Keras CNN found on another repository of my github account (over-writing the initial data loading). Performance declines from 69% accuracy to 64% accuracy, unfortunately, and expectedly.
