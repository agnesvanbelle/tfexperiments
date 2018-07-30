from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  '''
  the -1 signifies that the batch_size dimension will be dynamically calculated based on the 
  number of examples in our input data.'''
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  
  '''
  apply 32 5x5 filters to the input layer.
  Our output tensor produced by conv2d() has a shape of [batch_size, 28, 28, 32].'''
  conv1 = tf.layers.conv2d( inputs = input_layer,
                            filters = 32,
                            kernel_size=[5,5],
                            padding="same",
                            activation = tf.nn.relu)
  ''' 
  Our output tensor produced by max_pooling2d() (pool1) has a shape of [batch_size, 14, 14, 32]: 
  the 2x2 filter reduces width and height by 50% each.'''
  pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=[2,2], strides=2)
  
  conv2 = tf.layers.conv2d(inputs = pool1, 
                           filters = 64, 
                           kernel_size=[5,5],
                           padding = "same",
                           activation = tf.nn.relu)
  '''
  pool2 has shape [batch_size, 7, 7, 64] (50% reduction of width and height from conv2).'''
  pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size=[2,2], strides=2)
  
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu)
  dropout = tf.layers.dropout(inputs = dense, rate = 0.4, training = mode == tf.estimator.ModeKeys.TRAIN)
  logits = tf.layers.dense(inputs = dropout, units = 10)
  
  predictions = {
    "classes": tf.argmax(input = logits, axis=1, name="classes"), # our logits tensor has shape [batch_size, 10]
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)
  
  #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  #loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
  loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train_op = optimizer.minimize(loss, global_step= tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
  
  # mode is eval
  eval_metrics_ops = {
    "accuracy": tf.metrics.accuracy(labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metrics_ops)
  
def main(argv, train=False, test=True):
  print("hi")
  print(argv)
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  
  mnist_classifier = tf.estimator.Estimator(model_fn = cnn_model_fn, model_dir = "../../../data/mnist_model")
  tensors_to_log = {"my classes" : "classes"}
  logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter = 50)
  if train:
    train_input_fn  = tf.estimator.inputs.numpy_input_fn (
      x = {"x": train_data},
      y = train_labels,
      batch_size = 100,
      num_epochs = None,
      shuffle=True)
    mnist_classifier.train( input_fn = train_input_fn, 
                            steps = 1000,
                            hooks = [logging_hook])
  if test:
    eval_input_fn = tf.estimator.inputs.numpy_input_fn (
      x = {"x": eval_data},
      y = eval_labels,
      num_epochs = 1,
      shuffle = False
      )
    eval_results = mnist_classifier.evaluate(input_fn = eval_input_fn)
    print('eval results:', eval_results)
  
if __name__=="__main__":
  tf.app.run(main)
  
  
  
  
  