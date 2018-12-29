import sys, os

import pickle

def open_batch(fn):
  with open(fn, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

fn = 'data/cifar10/data_batch_1'

d = open_batch(fn)

print(d.keys())

print(d[b'data'].shape)