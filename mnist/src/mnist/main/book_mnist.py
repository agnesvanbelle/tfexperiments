import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


DATA_DIR = '../../../data/'
NUM_STEPS = 10000
MINIBATCH_SIZE = 100

writer = tf.summary.FileWriter('../../../data/graphs/book_mnist')


data = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784], "x") # 28 x 28 = 784
W = tf.get_variable("W", initializer=tf.zeros([784, 10]))
y_pred = tf.matmul(x, W)
y_true = tf.placeholder(tf.float32, [None, 10]) # "None" means we're not currently saying how many samples we will provide


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels =  y_true))
gd_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

writer.add_graph(tf.get_default_graph())

with tf.Session(graph=tf.get_default_graph()) as sess:
  sess.run(tf.global_variables_initializer())
  for _ in range(NUM_STEPS):
    batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
    sess.run(gd_step, feed_dict = {x: batch_xs, y_true: batch_ys})
  
  ans = sess.run(accuracy, feed_dict = {x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))