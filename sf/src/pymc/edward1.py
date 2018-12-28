import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import edward as ed
from edward.models import Normal

# https://github.com/mikkokemppainen/Jupyter_notebooks/blob/master/Edward_notebook_public.ipynb

x_train = np.linspace(-3, 3, num=50)
y_train = np.cos(x_train) + np.random.normal(0, 0.1, size=50)
x_train = x_train.astype(np.float32).reshape((50, 1))
y_train = y_train.astype(np.float32).reshape((50, 1))

print("x train shape", x_train.shape)
print("y train shape", y_train.shape)

plt.scatter(x_train, y_train)
#plt.show()


def neural_network(x_train, W_0, W_1, b_0, b_1):
  x = x_train # 50 rows, 1 column
  y = tf.matmul(x, W_0) + b_0 
  y = tf.tanh(y)
  y = tf.matmul(y, W_1) + b_1 
  return y

# init layers
W_0 = Normal(loc=tf.zeros([1, 2]), scale=tf.ones([1, 2])) # W_0 has 1 row, 2 columns
W_1 = Normal(loc=tf.zeros([2, 1]), scale=tf.ones([2, 1])) # W_1 has 2 rows, 1 column
b_0 = Normal(loc=tf.zeros(2), scale=tf.ones(2))
b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1))

print("NN shape: %s" % "x".join(str(x) for x in [x_train.shape, W_0.shape, W_1.shape]))

# make NN
y = Normal(loc = neural_network(x_train, W_0, W_1, b_0, b_1), scale = 0.1) # make NN
print("y:",y)

# specify a normal approximation over the weights and biases
qW_0 = Normal(tf.get_variable("qW_0/loc", [1,2]), scale = tf.nn.softplus(tf.get_variable('qW_0/scale', [1,2])))
qW_1 = Normal(tf.get_variable("qW_1/loc", [2,1]), scale = tf.nn.softplus(tf.get_variable('qW_1/scale', [2,1])))
qb_0 = Normal(loc=tf.get_variable("qb_0/loc", [2]), scale=tf.nn.softplus(tf.get_variable("qb_0/scale", [2])))
qb_1 = Normal(loc=tf.get_variable("qb_1/loc", [1]), scale=tf.nn.softplus(tf.get_variable("qb_1/scale", [1])))


# run variational inference
inference = ed.KLqp({W_0: qW_0, b_0:qb_0, W_1: qW_1, b_1: qb_1}, data={y: y_train})
print("inference object:", inference)
res = inference.run(n_iter=1000)

n_samples = 1000


qW_0_samples = qW_0.sample(sample_shape=n_samples)
qW_1_samples = qW_1.sample(sample_shape=n_samples)
qb_0_samples = qb_0.sample(sample_shape=n_samples)
qb_1_samples = qb_1.sample(sample_shape=n_samples)

print("W_) samples:", qW_0_samples)







