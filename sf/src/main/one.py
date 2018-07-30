import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

a = tf.add(3,5)

with tf.Session() as sess:

  return_value = sess.run(a)


print ("return value:", return_value)

###

x = 2
y = 3
add_op = tf.add(x,y)
mul_op = tf.multiply(x,y)
useless = tf.multiply(x,add_op)
pow_op = tf.pow(add_op, mul_op)
with tf.Session() as sess:
  print(sess.run([useless, pow_op]))
  
  
###


g = tf.Graph()

with g.as_default():
  x = tf.add(3, 5)
x2 = tf.add(4,5)
with tf.Session(graph=g) as sess:
  print(sess.run(x))
with tf.Session(graph = tf.get_default_graph()) as sess:
  print(sess.run(x2))
