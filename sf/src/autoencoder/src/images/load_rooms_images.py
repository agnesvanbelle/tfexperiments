import skimage
import os
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

#folder = "/data/users/vanbelle/pic2/images/"
THIS_DIR = os.path.dirname(__file__)
ROOMS_FOLDER = os.path.join(THIS_DIR, "../../data/images_sample/")
MAX_IMG = 1000
MAX_WIDTH, MAX_HEIGHT = 640, 640 # find_max_min_size()[0][0], find_max_min_size()[0][1]
RESIZE_DIM = 64#128 #256

def find_max_min_size():
  max_shape = (0,0,0) # 640, 640, 3
  min_shape = (sys.maxsize, sys.maxsize, sys.maxsize)
  i = 0
  for f in os.listdir(ROOMS_FOLDER):
    subfolder = os.path.join(ROOMS_FOLDER, f)
    if os.path.isfile(subfolder):
      continue
    for f in os.listdir(subfolder):
      full_path = os.path.join(subfolder, f)
     
      if not os.path.isfile(full_path):
        continue
      img = skimage.io.imread(full_path)
      i += 1
      if i > MAX_IMG:
        break
      if i % 100 == 0:
        print("processed {:d} images".format(i))
        
      max_shape = max(img.shape, max_shape)
      min_shape = min(img.shape, min_shape)
  return max_shape, min_shape


def make_symmetric(img):
  max_dim = max(MAX_WIDTH, MAX_HEIGHT)
  largest_dim = np.argmax(img.shape)
  scale_factor = max_dim / img.shape[largest_dim]
  scale_1 = max(max_dim, int(img.shape[0] * scale_factor))
  scale_2 = max(max_dim, int(img.shape[1] * scale_factor))
  img = skimage.transform.resize(img, (scale_1, scale_2, 3), mode='edge', cval=0.5)
  
  nr_pad_1 = int((max_dim - scale_1) / 2)
  nr_pad_2 = int((max_dim - scale_2) / 2)    
  img = np.pad(img, ((nr_pad_1, nr_pad_1), (nr_pad_2, nr_pad_2), (0,0)), mode='constant')
      
  return img

def load_rooms_images():
  l = []
  i = 0
  for f in os.listdir(ROOMS_FOLDER):
    subfolder = os.path.join(ROOMS_FOLDER, f)
    if os.path.isfile(subfolder):
      continue
    for f in os.listdir(subfolder):
      full_path = os.path.join(subfolder, f)
     
      if not os.path.isfile(full_path):
        continue
      img = skimage.io.imread(full_path)
        
      img = make_symmetric(img)
      img = skimage.transform.resize(img, (RESIZE_DIM, RESIZE_DIM, 3), mode='edge', cval=0.5)
      
      i += 1
      if i > MAX_IMG:
        break
      if i % 100 == 0:
        print("processed {:d} images".format(i))
      #print(img.shape)
      #plt.imshow(img[:,:,1])
      #plt.imshow(img)
      #plt.show()
      l.append(img)
      #print(np.max(img), np.min(img))
  return np.array(l)



