import sys, os
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import tensorflow.contrib.layers as lays


def autoencoder(inputs):
  # encoder
  # 
  l1 = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
  print("net after l1:", l1)
  l2 = lays.conv2d(l1, 16, [5, 5], stride=2, padding='SAME')
  print("net after l2:", l2)
  l3 = lays.conv2d(l2, 8, [5, 5], stride=4, padding='SAME')
  print("net after l3:", l3)
  # decoder
  l4 = lays.conv2d_transpose(l3, 16, [5, 5], stride=4, padding='SAME')
  print("net after l4:", l4)
  l5 = lays.conv2d_transpose(l4, 32, [5, 5], stride=2, padding='SAME')
  print("net after l5:", l5)
  l6 = lays.conv2d_transpose(l5, 3, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid)#tanh)
  print("net after l6:", l6)
  return l3, l6

def train(stacked_images, nr_epochs = 100, batch_size = 0.1):
  BATCH_SIZE_MIN = 100
  
  nr_images, img_height, img_width = stacked_images.shape[0], stacked_images.shape[1], stacked_images.shape[2]
  ae_inputs = tf.placeholder(tf.float32, (None, img_height, img_width, 3), name="inputs")  # input to the network (images)
  
  feature_layer, ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network
  tf.summary.histogram("feature_layer histogram", feature_layer)
  # calculate the loss and optimize the network
  loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # calculate the mean square error loss
  tf.summary.scalar("L2 loss", loss) 
  
  train_op = tf.train.AdamOptimizer(learning_rate=.01).minimize(loss, global_step=tf.train.create_global_step())

  # initialize the network
  init = tf.global_variables_initializer()
  
  nr_train = int(0.9 * nr_images)
  train_part = stacked_images[:nr_train,]
  test_part = stacked_images[nr_train:,]
  print("train data shape: {:}, test data shape: {:}".format(train_part.shape, test_part.shape))
  
  batch_size = max(BATCH_SIZE_MIN, batch_size) if batch_size > 1 else int(max(BATCH_SIZE_MIN, nr_images * batch_size))
  print("batch size:", batch_size)
  
  merged_summaries = tf.summary.merge_all()
  
  writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
  
  with tf.Session() as sess:
    sess.run(init)
    for epoch_iter in range(nr_epochs):
        for batch_start_index in range(0, nr_train, batch_size): 
          batch_nr = int(batch_start_index / batch_size)
          train_batch = train_part[batch_start_index:batch_start_index + batch_size, :]
          #print ("train batch shape: ", train_batch.shape)
          _, summ, c = sess.run([train_op, merged_summaries, loss], feed_dict = {ae_inputs: train_batch})
          print('Epoch: {}, batch {}, step {}, - cost= {:.5f}'.format((epoch_iter + 1), batch_nr + 1, sess.run(tf.train.get_global_step()), c))
  
    reconstructed_images = sess.run([ae_outputs], feed_dict={ae_inputs: train_part})[0]
    
    print(reconstructed_images.shape)
    print(type(reconstructed_images))
  
  writer.close()
  
  for i in range(reconstructed_images.shape[0]):
    img_orig = train_part[i,:]
    img_recon = reconstructed_images[i,:]
    print(img_recon.shape)
    print("dim1:", img_recon[:,:,0])
    print("dim2:", img_recon[:,:,1])
    print("dim3:", img_recon[:,:,2])
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(img_orig)
    fig.add_subplot(1, 2, 2)
    plt.imshow(img_recon)
    #plt.imshow(img)
    plt.show()
