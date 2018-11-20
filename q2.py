# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:30:08 2018

@author: brand
"""

from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os.path as path
from sklearn.datasets import fetch_mldata

##Getting the raw data from the database
datafile = "mnist-original.mat"
if(not path.isfile(datafile)):
    try:
        mnist = fetch_mldata('MNIST original')
    except: #Implemented a work around since mldata.org is down
        print("Could not download MNIST data from mldata.org, trying alternative...")
        from six.moves import urllib
        from scipy.io import loadmat
        mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
        mnist_path = "./mnist-original.mat"
        response = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_path, "wb") as f:
            content = response.read()
            f.write(content)
        mnist_raw = loadmat(mnist_path)
        print("Success!")
        
#Used to display images throughout
def display(data, label):
    pixels = np.array(data, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    plt.title(label)
    plt.imshow(pixels, cmap='Greys')
    plt.show()
    
#Creating the subsets
mnist = {"data": mnist_raw["data"].T,"labels": mnist_raw["label"][0]}
data = mnist['data']
labels = mnist['labels']

dist = np.array([])

for i in range(9,14):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
    dist = np.append(dist,kmeans.inertia_)

#### RBG Network
#def gaussAct(x):
#    b = 0.5
#    c = 
#    return tf.math.exp(-0.5*(x-))

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h1, w_o):
    h = tf.nn.gaussian(X)
    #h1 = tf.nn.sigmoid(tf.matmul(X, w_h1)) # this is a basic mlp, think 2 stacked logistic regressions    
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

size_h1 = tf.constant(625, dtype=tf.int32)
size_h2 = tf.constant(300, dtype=tf.int32)

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h1 = init_weights([784, size_h1]) # create symbolic variables
w_h2 = init_weights([size_h1, size_h2])
w_o = init_weights([size_h2, 10])

py_x = model(X, w_h1, w_h2, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    print(range(0,len(trX),128))
    for i in range(3):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        pred = sess.run(py_x, feed_dict={X: trX})
        print(i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX})))
    saver.save(sess,"mlp/session.ckpt")
    