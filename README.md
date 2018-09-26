# Tensorflow implementation of digit recognizer on MNIST data
MNIST's handwritten digit database is a very basic dataset for learning neural networks. The dataset contains 70,000 images of 28x28 pixels size. The implementation uses Tensorflow library for creating CNNs. 

There are two implementations.
* using tf.estimator
* using low-level tensorflow APIs

*There are proper comments on the flow of the code*

### tf.estimator
tf.estimator is a high-level API which encapsulates the training, evaluation and prediction aspects of a ML model. The estimator object takes in the model function which describes the layer architechture of the CNN. The user can specify
* number of convulutional layers and its specifics
* number of pooling layers and its specifics
* number of dense layers and its specifics
* type of optimiser 
* evaluation metrics

There is a logging_hook to check the progress of your training via console. 

### low-level APIs
We can also implement CNNs without using estimators. This is very similar to the way we write the model function. 



