import tensorflow as tf
from edward.models import Normal
import numpy as np
import matplotlib.pyplot as plt
import edward as ed


# https://github.com/mikkokemppainen/Jupyter_notebooks/blob/master/Edward_notebook_public.ipynb

x_train = np.linspace(-3, 3, num=50)
y_train = np.cos(x_train) + np.random.normal(0, 0.1, size=50)
x_train = x_train.astype(np.float32).reshape((50, 1))
y_train = y_train.astype(np.float32).reshape((50, 1))

print(x_train)
print(y_train)

plt.scatter(x_train, y_train)
#plt.show()
W_0 = Normal(loc=tf.zeros([1, 2]), scale=tf.ones([1, 2]))
W_1 = Normal(loc=tf.zeros([2, 1]), scale=tf.ones([2, 1]))
b_0 = Normal(loc=tf.zeros(2), scale=tf.ones(2))
b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1))

print(W_0)
print(tf.zeros([1,2]))

print(np.zeros([1,2]))

def neural_network(x_train, W_0, W_1, b_0, b_1):
  x = x_train # 50 rows, 1 column
  y = tf.matmul(x, W_0) + b_0 # 1 row, 2 columns
  y = tf.tanh(y)
  y = tf.matmul(y, W_1) + b_1 # W_1 has 2 rows, 1 column
  y = Normal(loc=y, scale=0.1)
  return y



# specify a normal approximation over the weights ans biases
qW_0 = Normal(tf.get_variable("qW_0/loc", [1,2]), scale = tf.nn.softplus(tf.get_variable('qW_0/scale', [1,2])))
qW_1 = Normal(tf.get_variable("qW_1/loc", [2,1]), scale = tf.nn.softplus(tf.get_variable('qW_1/scale', [2,1])))
qb_0 = Normal(loc=tf.get_variable("qb_0/loc", [2]), scale=tf.nn.softplus(tf.get_variable("qb_0/scale", [2])))
qb_1 = Normal(loc=tf.get_variable("qb_1/loc", [1]), scale=tf.nn.softplus(tf.get_variable("qb_1/scale", [1])))

# run variational inference

y = neural_network(x_train, W_0, W_1, b_0, b_1)
inference = ed.KLqp({W_0: qW_0, b_0:qb_0, W_1: qW_1, b_1: qb_1}, data={y: y_train})
print(inference)
res = inference.run(n_iter=1000)

n_samples = 1000


qW_0_samples = qW_0.sample(sample_shape=n_samples)
qW_1_samples = qW_1.sample(sample_shape=n_samples)
qb_0_samples = qb_0.sample(sample_shape=n_samples)
qb_1_samples = qb_1.sample(sample_shape=n_samples)

print(qW_0_samples)







