import tensorflow as tf
import skimage
import os
import matplotlib
from matplotlib import pyplot as plt
import sys
from skimage import transform
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from PIL import Image
import tensorflow.contrib.layers as lays



#folder = "/data/users/vanbelle/pic2/images/"
folder = "data/images_sample/"

max_img = 1000

def find_max_min_size():
  max_shape = (0,0,0) # 640, 640, 3
  min_shape = (sys.maxsize, sys.maxsize, sys.maxsize)
  i = 0
  for f in os.listdir(folder):
    subfolder = os.path.join(folder, f)
    if os.path.isfile(subfolder):
      continue
    for f in os.listdir(subfolder):
      full_path = os.path.join(subfolder, f)
     
      if not os.path.isfile(full_path):
        continue
      img = skimage.io.imread(full_path)
     
      max_shape = max(img.shape, max_shape)
      min_shape = min(img.shape, min_shape)
  return max_shape, min_shape

max_width, max_height = 640, 640 # find_max_min_size()[0][0], find_max_min_size()[0][1]

print("max width:", max_width)
print("max height:", max_height)

resize_to_size = 128 #256

def get_image_list():
  l = []
  i = 0
  for f in os.listdir(folder):
    subfolder = os.path.join(folder, f)
    if os.path.isfile(subfolder):
      continue
    for f in os.listdir(subfolder):
      full_path = os.path.join(subfolder, f)
     
      if not os.path.isfile(full_path):
        continue
      img = skimage.io.imread(full_path)
         
      i += 1
      if i > max_img:
        break
      if i % 100 == 0:
        print("processed {:d} images".format(i))
        
      max_dim = max(max_width, max_height)
      largest_dim = np.argmax(img.shape)
      scale_factor = max_dim / img.shape[largest_dim]
      scale_1 = max(max_dim, int(img.shape[0] * scale_factor))
      scale_2 = max(max_dim, int(img.shape[1] * scale_factor))
      img = skimage.transform.resize(img, (scale_1, scale_2, 3), mode='edge', cval=0.5)
      
      nr_pad_1 = int((max_dim - scale_1) / 2)
      nr_pad_2 = int((max_dim - scale_2) / 2)    
      img = np.pad(img, ((nr_pad_1, nr_pad_1), (nr_pad_2, nr_pad_2), (0,0)), mode='constant')
      
      img = skimage.transform.resize(img, (resize_to_size, resize_to_size, 3), mode='edge', cval=0.5)
      
      #print(img.shape)
      #plt.imshow(img[:,:,1])
      #plt.imshow(img)
      #plt.show()
      l.append(img)
      #print(np.max(img), np.min(img))
  return l


l = get_image_list()
stacked_images = np.array(l)

print("stacked images shape:", stacked_images.shape)
# 
# https://github.com/giuseppebonaccorso/lossy_image_autoencoder/blob/master/Lossy%20Image%20Autoencoder.ipynb
# https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow
# http://cs231n.github.io/convolutional-networks/
# https://mourafiq.com/2016/08/10/playing-with-convolutions-in-tensorflow.html
# https://github.com/mr-ravin/CNN-Autoencoders/blob/master/script.py
def autoencoder(inputs):
  # encoder
  # 
  l1 = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
  print("net after l1:", l1)
  l2 = lays.conv2d(l1, 16, [5, 5], stride=2, padding='SAME')
  print("net after l2:", l2)
  l3 = lays.conv2d(l2, 8, [5, 5], stride=4, padding='SAME')
  print("net after l3:", l3)
  # decoder
  l4 = lays.conv2d_transpose(l3, 16, [5, 5], stride=4, padding='SAME')
  print("net after l4:", l4)
  l5 = lays.conv2d_transpose(l4, 32, [5, 5], stride=2, padding='SAME')
  print("net after l5:", l5)
  l6 = lays.conv2d_transpose(l5, 3, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
  print("net after l6:", l6)
  return l3, l6


ae_inputs = tf.placeholder(tf.float32, (None, resize_to_size, resize_to_size, 3))  # input to the network (images)
feature_layer, ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network
# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # calculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=.001).minimize(loss)
# initialize the network
init = tf.global_variables_initializer()

nr_train = int(0.9 * stacked_images.shape[0])
train_part = stacked_images[:nr_train,]
test_part = stacked_images[nr_train:,]
print("train data shape: {:}, test data shape: {:}".format(train_part.shape, test_part.shape))

nr_epochs = 14

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
  sess.run(init)
  for epoch_iter in range(nr_epochs):
    _, c = sess.run([train_op, loss], feed_dict = {ae_inputs: train_part})
    print('Epoch: {} - cost= {:.5f}'.format((epoch_iter + 1), c))

  reconstructed_images = sess.run([ae_outputs], feed_dict={ae_inputs: train_part})[0]
  
  print(reconstructed_images.shape)
  print(type(reconstructed_images))

writer.close()

for i in range(reconstructed_images.shape[0]):
  img_orig = train_part[i,:]
  img_recon = reconstructed_images[i,:]
  print(img_recon.shape)
  print(np.max(img_recon), np.min(img_recon))
  fig = plt.figure()
  fig.add_subplot(1, 2, 1)
  plt.imshow(img_orig)
  fig.add_subplot(1, 2, 2)
  plt.imshow(img_recon)
  #plt.imshow(img)
  plt.show()

