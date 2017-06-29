from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("Mnist_data/",one_hot=True)

#=========================================================================================================
# Reminder :The MNIST dataset is a set of greyscale images with 28*28=784 pixels. Each image is a number
# between 0 and 9. The goal of the practical is to obtain a classifier that can correctly classify the
# images into the 10 classes. The MNIST data consists of 55000 train images and labels, 5000 observations
#  validation set and 10000 observations in the test set.
#=========================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,[None,784])  # input
y_ = tf.placeholder(tf.float32,[None,10])  # labels

#initialization weights and biases
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#convolution and pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# first convolution
W_conv1 = weight_variable([5, 5, 1, 32])  #batch size 5x5, 1 input, 32 output
b_conv1 = bias_variable([32])             #each output have a special biases

x_image = tf.reshape(x, [-1,28,28,1])     # 2 weight 3 high 4 color (1 grey, 3 rgb)

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolution
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2=max_pool_2x2(h_conv2)


#
W_fc=weight_variable([7*7*64,1024])
b_fc=bias_variable([1024])
x_fc=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(x_fc, W_fc) + b_fc)

W_fcm = weight_variable([1024, 10])
b_fcm = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fcm) + b_fcm)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(10):
  batch = mnist.train.next_batch(100)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1]})
    #print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})



print("test accuracy %g" % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# a simple visualisation for the numbers where we classify. For watching, copy all the code in this programme in ipython notebook
#feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
#inferred_labels = tf.argmax(y_conv, 1).eval(feed_dict=feed_dict)
#correct_labels = tf.argmax(y_, 1).eval(feed_dict=feed_dict)
#is_correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1)).eval(feed_dict=feed_dict)
#misclassified_idxs = np.arange(len(mnist.test.images))[~is_correct]
#plt.figure(figsize=(18, 8))
#subplot = 1
#plt.show()
#for i in np.random.choice(misclassified_idxs, size=10, replace=False):
#    plt.subplot(2, 6, subplot)
#    plt.imshow(mnist.test.images[i].reshape(28, 28))

#    subplot += 1
#    title = 'Classified as {} should be {}'.format(inferred_labels[i],
                                                   correct_labels[i])
#    plt.title(title)
#    plt.show()
