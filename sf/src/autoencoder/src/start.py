import sys, os

from autoencoder.src.images import load_cifar_images, load_rooms_images
from autoencoder.src.models import encoder1


## resources:
# https://github.com/giuseppebonaccorso/lossy_image_autoencoder/blob/master/Lossy%20Image%20Autoencoder.ipynb
# http://machinelearninguru.com/deep_learning/tensorflow/neural_networks/autoencoder/autoencoder.html
# https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c
# https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow
# http://cs231n.github.io/convolutional-networks/
# https://mourafiq.com/2016/08/10/playing-with-convolutions-in-tensorflow.html
# https://github.com/mr-ravin/CNN-Autoencoders/blob/master/script.py
# https://github.com/arashsaber/Deep-Convolutional-AutoEncoder/blob/master/ConvolutionalAutoEncoder.py
# https://towardsdatascience.com/autoencoder-for-converting-an-rbg-image-to-a-gray-scale-image-3c19a11031c9
# https://k-d-w.org/blog/103/denoising-autoencoder-as-tensorflow-estimator
# https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/

imgs_cifar = load_cifar_images.load_cifar_images()
#imgs_rooms = load_rooms_images.load_rooms_images()

encoder1.train(imgs_cifar) 
'''
encoder1.train(imgs_cifar) was good with nr_epochs = 100, batch_size = 0.1:
Epoch: 100, batch 1, - cost= 0.01458
Epoch: 100, batch 2, - cost= 0.01373
Epoch: 100, batch 3, - cost= 0.01432
Epoch: 100, batch 4, - cost= 0.01376
Epoch: 100, batch 5, - cost= 0.01437
Epoch: 100, batch 6, - cost= 0.01444
Epoch: 100, batch 7, - cost= 0.01415
Epoch: 100, batch 8, - cost= 0.01396
'''