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

folder = "/data/users/vanbelle/pic2/images/"

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

resize_to_size = 32

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
  return l


l = get_image_list()
stacked_images = np.array(l)

print(stacked_images.shape)
# 
# 
# https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow
# http://cs231n.github.io/convolutional-networks/
# https://mourafiq.com/2016/08/10/playing-with-convolutions-in-tensorflow.html
# https://github.com/mr-ravin/CNN-Autoencoders/blob/master/script.py
def autoencoder(inputs):
  # encoder
  # 32 x 32 x 1   ->  16 x 16 x 32
  # 16 x 16 x 32  ->  8 x 8 x 16
  # 8 x 8 x 16    ->  2 x 2 x 8
  net = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
  print(net)
  net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
  net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
  # decoder
  # 2 x 2 x 8    ->  8 x 8 x 16
  # 8 x 8 x 16   ->  16 x 16 x 32
  # 16 x 16 x 32  ->  32 x 32 x 1
  net = lays.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
  net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
  net = lays.conv2d_transpose(net, 3, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
  return net


ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 3))  # input to the network (images)
ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network
# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=.01).minimize(loss)
# initialize the network
init = tf.global_variables_initializer()

nr_train = int(0.9 * max_img)
train_part = stacked_images[:nr_train,]
test_part = stacked_images[nr_train:,]
print(train_part.shape, test_part.shape)

nr_epochs = 500

with tf.Session() as sess:
  sess.run(init)
  for epoch_iter in range(nr_epochs):
    _, c = sess.run([train_op, loss], feed_dict = {ae_inputs: train_part})
    print('Epoch: {} - cost= {:.5f}'.format((epoch_iter + 1), c))

  reconstructed_images = sess.run([ae_outputs], feed_dict={ae_inputs: train_part})[0]
  
  print(reconstructed_images.shape)
  print(type(reconstructed_images))
  
for i in range(reconstructed_images.shape[0]):
  img = reconstructed_images[i,:]
  print(img.shape)
  plt.imshow(img)
  #plt.imshow(img)
  plt.show()

