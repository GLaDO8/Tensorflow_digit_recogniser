import tensorflow as tf
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#data fetch from sklearn database
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#placeholder variables. here, None represents the size of the batch, which can be anything
x = tf.placeholder(tf.float32, shape=[None, 784], name = "28X28_flat_input")
y_oh = tf.placeholder(tf.float32, shape=[None, 10], name = "one-hot output")

#input layer
input_layer = tf.reshape(x, [-1, 28, 28, 1]) #reshape(tensor, [batch_size, image_height, image_width, RGB_channels])

#convolution layer 1
convoL1 = tf.layers.conv2d(
    inputs = input_layer,
    filters = 32,
    kernel_size = 5,
    padding = "same",
    activation = tf.nn.relu,
    use_bias = True,
    bias_initializer = None,
    kernel_initializer = None
)
#note: if kernel is height and width are same, you can pass an integerself. output size - [batch_size, 28, 28, 32]

#pooling layer 1
poolL1 = tf.layers.max_pooling2d(
    inputs = convoL1,
    pool_size = 2,
    strides = 2
)
#note: output size - [batch_size, 28, 28, 32]

#covolution layer 2
convoL2 = tf.layers.conv2d(
    inputs = poolL1,
    filters = 64,
    kernel_size = 5,
    padding = "same",
    activation = tf.nn.relu,
    use_bias = True,
    bias_initializer = None,
    kernel_initializer = None
)
#note output size - [batch_size, 7, 7, 64]

#pooling layer 2
poolL2 = tf.layers.max_pooling2d(
    inputs = convoL2,
    pool_size = 2,
    strides = 2
)

#flattening
poolL2_flat = tf.reshape(poolL2, [-1, 7*7*64])

#dense layer 1
denseL1 = tf.layers.dense(
    inputs = poolL2_flat,
    units = 1024,
    activation = tf.nn.relu,
    use_bias = True,
    bias_initializer = None,
    kernel_initializer = None
)

#dropout regularisation (rate = 0.4)
dropoutL1 = tf.layers.dropout(
    inputs = denseL1,
    rate = 0.4,
    training = (mode == tf.estimator.ModeKeys.TRAIN)
)

#dense layer 2
logits = tf.layers.dense(
    inputs = dropoutL1,
    units = 10
)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_oh, logits = logits))

#The optimizer has a parameter called var_lists. var_lists takes in a tuple or list of Variables to update to minimise loss. If not mentioned, it takes in variables which are under the key "GraphKeys.TRAINABLE_VARIABLES"
training_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#prediction of our model. here tf.argmax takes in the input tensor and the axis of measurement to give the max value. tf.equal will compare and give a tensor of booleans
prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#tf.cast will convert the boolean to a float between 0 and 1. reduce_mean will compute the
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

for _ in range(20000):
    input, label = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))

    training_step.run(feed_dict={x: input, y_oh: label})

#print the accuracy of the model
print("test accuracy %g" %accuracy.eval(feed_dict={x: mnist.test.images, y_oh: mnist.test.labels, keep_prob: 1.0}))
