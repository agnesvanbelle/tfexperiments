import tensorflow as tf
import numpy as np
import sys

# b = tf.get_variable("t", initializer = tf.constant([10, 20, 30, 40, 50, 60, 70]))
# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   result = sess.run(tf.reduce_mean(b))
#   print(result)
#   
  
train_x = np.transpose(np.array([np.linspace(-1, 1, 11), np.linspace(-2, 2, 11)]))
train_y = 3 * train_x[:,0] + np.random.randn(*train_x[:,0].shape) * 0.33
print(train_x)
print(train_y)

#sys.exit(0)
X = tf.placeholder(tf.float32, shape = [2,], name="x")
y = tf.placeholder(tf.float32, shape = [], name="y")

w = tf.get_variable("weights", shape=[2], initializer = tf.zeros_initializer(tf.float32))

y_model = tf.multiply(X, w, name = "multiply")

cost = tf.pow(y - y_model, 2)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train_op = optimizer.minimize(cost, name = "minimize")

my_graph = tf.get_default_graph()
writer = tf.summary.FileWriter('./graphs')
writer.add_graph(my_graph)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(100):
    for (t_x, t_y) in zip(train_x, train_y):
      #print(t_x.shape)
      sess.run(train_op, feed_dict = {X : t_x, y: t_y})
  print(sess.run(w))

writer.close()

# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   print(sess.run(w))
  
  