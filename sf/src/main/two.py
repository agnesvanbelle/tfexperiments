import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.constant(3, name='a')
b = tf.constant(5, name='b')
x = tf.add(a,b, name='add')

# Create the summary writer after graph definition and before running your session
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
  print(sess.run(x))
writer.close()

my_graph = tf.Graph()

with my_graph.as_default():
  a2 = tf.constant([2,2], name='a2')
  b2 = tf.constant([[0,1],[2,3]], name='b2')
  x2 = tf.multiply(a2, b2, name='mult')
  z = tf.zeros([2, 3], tf.int32) 
  f = tf.fill([2, 3], 8)
  c = tf.add(z,f)
writer = tf.summary.FileWriter('./graphs2')
writer.add_graph(my_graph)
with tf.Session(graph=my_graph) as sess:
  res1, res2 = sess.run([c, x2])
  print (res1, type(res1))
  print (res2, type(res2))
writer.close()

##########

my_const = tf.constant([3.0, 2.0])

s = tf.get_variable("myscalar", initializer=tf.constant(2)) 
m = tf.get_variable("mymatrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("mybig_matrix", shape=(784, 10), initializer=tf.zeros_initializer())
assign_op = W.assign(tf.random_uniform(W.shape))
with tf.Session() as sess:
  #print(sess.graph.as_graph_def())
  #sess.run(tf.global_variables_initializer())
  #print("W:",sess.run(W))
  sess.run(W.initializer)
  sess.run(assign_op)
  print(W.eval())

#######

my_var = tf.get_variable("my_var", initializer=tf.constant(2))


  


