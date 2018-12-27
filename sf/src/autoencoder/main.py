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

folder = "/data/users/vanbelle/pic2/images/"

max_img = 14
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
    
    i += 1
    if i > max_img:
      break
    
    largest_dim = np.argmax(img.shape)
    scale_factor = 640 / img.shape[largest_dim]
    scale_1 = int(img.shape[0] * scale_factor)
    scale_2 = int(img.shape[1] * scale_factor)
    img = skimage.transform.resize(img, (scale_1, scale_2, 3), mode='edge', cval=0.5)
    
    nr_pad_1 = int((640 - scale_1) / 2)
    nr_pad_2 = int((640 - scale_2) / 2)    
    img = np.pad(img, ((nr_pad_1, nr_pad_1), (nr_pad_2, nr_pad_2), (0,0)), mode='constant')

    print(img.shape)
    #plt.imshow(img[:,:,1])
    plt.imshow(img)
    plt.show()
    
print("max shape:", max_shape)
print("min shape:", min_shape)