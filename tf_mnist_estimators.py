#__future__ is used to make sure the imported modules work on older versions of python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.INFO)

#model function contains the following layers (inputL->convoL1->poolL1->convoL2->poolL2->denseL1->denseL2)
def mnist_model_func(features, labels, mode):

    #input layer
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1]) #reshape(tensor, [batch_size, image_height, image_width, RGB_channels])

    #convolution layer 1
    convoL1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = 5,
        padding = "same",
        activation = tf.nn.relu
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
        activation = tf.nn.relu
    )
    #note output size - [batch_size, 14, 14, 64]

    #pooling layer 2
    poolL2 = tf.layers.max_pooling2d(
        inputs = convoL2,
        pool_size = 2,
        strides = 2
    )
    #note output size - [batch_size, 7, 7, 64]

    #flattening
    poolL2_flat = tf.reshape(poolL2, [-1, 7*7*64])

    #dense layers
    denseL1 = tf.layers.dense(
        inputs = poolL2_flat,
        units = 1024,
        activation = tf.nn.relu
    )

    #dropout regularisation (rate = 0.4)
    dropoutL1 = tf.layers.dropout(
        inputs = denseL1,
        rate = 0.4,
        training = (mode == tf.estimator.ModeKeys.TRAIN)
    )
    #note - Training takes a boolean of whether the model is currently under training

    #logits layer
    logits = tf.layers.dense(
        inputs = dropoutL1,
        units = 10
    )

    #predictions dict (two types of predictions: 1) class based 2) probability based)
    predictions = {"classes": tf.argmax(
                        input = logits,
                        axis = 1),
                   "probabilities": tf.nn.softmax(
                        logits,
                        name = "softmax_tensor")}

    #compile all predictions in a EstimatorSpec object and return it
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions
        )

    #creates onehot style labels array for all the different label outputs.
    onehot_labels = tf.one_hot(indices = (tf.cast(labels, tf.int32)), depth = 10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels = onehot_labels,
        logits = logits
    )

    #training
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(
            mode = mode,
            loss = loss,
            train_op = train_op
        )

    #accuracy metrics
    eval_metric = {
        "accuracy": tf.metrics.accuracy(
            labels = labels,
            predictions = predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode = mode,
        loss = loss,
        eval_metric_ops = eval_metric
    )

def main(unused_argv):
    #loading MNIST data for training and evaluation
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images #np.array
    train_labels = np.asarray(mnist.train.labels, dtype = np.int32) #converts labels into an nparray
    eval_data = mnist.test.images #np.array
    eval_labels = np.asarray(mnist.test.labels, dtype = np.int32) #converts labels into an nparray

    #estimator creation
    MNIST_classifier = tf.estimator.Estimator(
        model_fn = mnist_model_func,
        model_dir = '/Users/glados/cnn'
    )

    #logging hook to check logs while the CNN runs
    tensors = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors = tensors,
        every_n_iter = 50
    )

    #training
    training_input_func = tf.estimator.inputs.numpy_input_fn(
        x = {"x": train_data},
        y = train_labels,
        batch_size = 100,
        num_epochs = None,
        shuffle = True
    )
    MNIST_classifier.train(
        input_fn = training_input_func,
        steps = 20000,
        hooks = ([logging_hook])
    )

    #evaluation
    eval_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": eval_data},
        y = eval_labels,
        num_epochs = 1,
        shuffle = False
    )
    # eval_result = MNIST_classifier.evaluate(input_fn = eval_fn)
    # print(eval_result)

    predict_func = 
    MNIST_classifier.predict(
        input_fn = training_input_func,

    )


if __name__ == "__main__":
    tf.app.run()
