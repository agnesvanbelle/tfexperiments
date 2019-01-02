import sys, os
import numpy as np
import pickle
from matplotlib import pyplot as plt
import math

THIS_DIR = os.path.dirname(__file__)
CIFAR_FN = os.path.join(THIS_DIR, '../../data/cifar10/data_batch_1')


def load_cifar_images():
  def open_cifar_batch(fn):
    with open(fn, 'rb') as fo:
      d = pickle.load(fo, encoding='bytes')
    return d
  
  d = open_cifar_batch(CIFAR_FN)
  img_data = d[b'data']
  
  print(img_data.shape)
  nr_images = img_data.shape[0]
  img_data = np.reshape(img_data, (nr_images, 3, 32, 32))
  img_data = np.transpose(img_data, (0,2,3,1))
  img_data = img_data.astype(np.float32)
  for i in range(nr_images):
    #print(img_data[i,:])
    img_data[i,:] = img_data[i,:] / float(255)
    #print(img_data[i,:])
  print(img_data.shape)
  return img_data
  
def show_cifar_imgs(img_data, number):
  fig = plt.figure()
  nrows = math.ceil(math.sqrt(number))
  ncols = math.ceil(math.sqrt(number))
  i = 0
  for i in range(number):
    fig.add_subplot(nrows, ncols,  i + 1)
    plt.imshow(img_data[i,:])
    i += 1
  plt.show()

if __name__ == "__main__":
  imgs = load_cifar_images()
  print(np.mean(imgs[0,:]))
  show_cifar_imgs(imgs, 25)

