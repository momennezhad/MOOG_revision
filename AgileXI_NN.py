
"""
Natural language processing libraries
"""
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Spacy Library

import spacy

# For the purpose of debugging

import pdb
import numpy as np
import random
import pickle
import timeit
from collections import Counter

# SKLearn metrics

from sklearn.metrics import accuracy_score

# Data download
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

mnist = input_data.read_data_sets("/tmp/data")

# Tensorflow model

import tensorflow as tf
sess = tf.Session()

# Input parameters

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

n_epochs = 500
batch_size = 100
learning_rate = 0.01

"""
Construction phase

The placeholder is used to represent the training data and the targets.

The shape of the X is only partiall defined. We know that it will be a 2D tensor, with instances 
along the first dimension and features along the second dimension, and we know that the number of features
is going to be ...

But we don't know yet how many instances each training batch will contain, so the shape.

We know that the y will be a 1D tensor with one entry per instance, but again we don't know the size of the training batch at this point, so the shape is None
"""

class make_NN(object):
    pass

def Functimer(func):
    """
    Timer for calculating how long the function the decorator 
    surrounds. 
    """
    start = timeit.timeit()
    def wrapper():
        func()
    end = timeit.timeit()
    t = end - start
    print t

# Placeholder variables

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name = "X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

"""
The actual neural network.

The placeholder x will act as the input layer, during the execution phase, it will be replaced 
with one training batch at a time.

One needs to create two hidden layers and the output layer

Creating the name_scope using the name of the layer - it will contain 
all the computation nodes for this neuron layer. This is optional, but the graph 
will look much nicer in TensorBoard.

"""
#@Functimer
def neuron_layer(X, n_neurons, name, activation=None):
    """
    Creating the neural network - the placeholder x will act as the input layer. 

    """
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)

        W = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name = "biases")
        z = tf.matmul(X,W) + b

        if activation=="relu":
            return tf.nn.relu(z)
        else:
            return z
        
hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
#hidden1 = neuron_layer(X,n_hidden1, "hidden1", activation_fn=tf.nn.elu)


hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
logits = neuron_layer(hidden2, n_outputs,"outputs")


    
# An automated version of this:
#with tf.name_scope("dnn"):
#    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
#    hidden2 = fully_connected(hidden2, n_hidden2, scope="hidden2")
#    logits = fully_connected(hidden2, n_outputs, scope="outputs",activation_fn=None)
           

"""
We have the neural network model, we have the cost function, and now we need to define a GradientDescentOptiizer 
that will tweak the model parameters to minimize the cost function. 
"""
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")

"""
The last important step in the construction phase to specify how to evaluate the model. We will simply 
use accuracy as our performance measure. 

First, for instance, determmine if the neural network's prediction is correct by checking whether 
"""

# TODO
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

"""
init initialized all variables 
"""

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            m,n = np.shape(X_batch)
            sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y:mnist.test.labels})
        print epoch, acc_train, acc_test
        
            
