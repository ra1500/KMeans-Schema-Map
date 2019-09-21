# KMeans-Schema-Map

Draws lines in an image between pixels of rare color in order for a convolutional neural network to better classify the objects. I call this a Schema Map Overlay. Applies KMeans Clustering of colors via the OpenCV API.

The 'Big Dipper' problem. If a CNN were to view a simple image of the night sky it might only declare that the objects of stars exist. It will be unable to provide the meaning of those stars, i.e. the constellation. To solve this and similar problems in computer vision I apply Schema Maps of various types/algorithms to preprocess images before they are put through a neural network. This one is a very simple example.

I applied it to the Keras CNN that I also have posted on github that uses the CIFAR10 image dataset. Unfortunately it performs poorly due to the small nature of the CIFAR10 images (32, 32, 3) which become overwhelmed with drawn lines.
