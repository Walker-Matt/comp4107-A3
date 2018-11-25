# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:30:08 2018

@author: brand
"""

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
from sklearn.datasets import fetch_mldata
import random
from matplotlib.pyplot import figure
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.neural_network import MLPClassifier

##Getting the raw data from the database
datafile = "mnist-original.mat"
mldata = True
if(path.isfile(datafile)):
    from scipy.io import loadmat
    mnist_path = "./" + datafile
    mnist_raw = loadmat(mnist_path)
    mldata = False
else:
    try:
        print("Downloading MNIST data from mldata.org")
        mnist_raw = fetch_mldata('MNIST original', one_hot=True)
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
        mldata = False
       
#Used to display images throughout
def display(data, label):
    pixels = np.array(data, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    plt.title(label)
    plt.imshow(pixels, cmap='Greys')
    plt.show()
    
#gaussian radial basis function
def gaussian(image,center,beta):
    return np.exp(-beta*(np.linalg.norm(image-center,ord=2)**2))

#randomly initiates weights from hidden layer to output
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

#Creating the subsets
if(mldata):
    mnist = {"data": mnist_raw["data"],"labels": mnist_raw["target"]}
else:
    mnist = {"data": mnist_raw["data"].T,"labels": mnist_raw["label"][0]}
data = mnist['data']

labels = np.array(mnist['labels'],dtype=int)
labels = np.vstack(labels)

#produces one-hot encoded arrays for labels
enc = OneHotEncoder(handle_unknown='ignore')
labels = enc.fit_transform(labels).toarray() 

numImages = 10000   #total number of images used for training/testing
indices = random.sample(range(len(data)), numImages)  #randomly grabs subsample of images
averages = np.array([])

#runs through k times with k-1 training and 1 testing group
kFold = np.array([10])
for k in kFold:
    size = int(numImages/k)  #number of images in each group
    groups = np.zeros((k,size,784))
    labs = np.zeros((k,size,10))
    
    #used for hidden layer neuron drop-out
    drop = np.random.choice([0,1], size = (25,))
    
    #splits the images/labels into k groups
    for i in range(k):
        groups[i] = data[indices[i*size:(i*size)+size]]
        labs[i] = labels[indices[i*size:(i*size)+size]]
    
    clusterCenters = []
    betaVals = []
    for i in range(k):
        print("Pass through #", i+1)
        train = np.zeros((size,784))
        trainLabels = np.zeros((size,10))
        test = groups[i]      #one group used for testing
        testLabels = labs[i]
        for j in range(k):
            #combined training groups into one
            if (not i==j):
                train = np.vstack((train,groups[j]))
                trainLabels = np.vstack((trainLabels,labs[j]))
        training = train[size:]
        trainLabels = trainLabels[size:]
        kmeans = KMeans(n_clusters=25).fit(training) #calculates kmeans using training images
        #kmeans = KMeans(n_clusters=25, random_state=0).fit(training) #calculates kmeans using training images
        clusterCenters.append(kmeans.cluster_centers_)  #coordinates of cluster centers
#        clusters.append(np.array([kmeans.cluster_centers_, kmeans.labels_, kmeans.inertia_]))
        
        #determines which points are in which cluster
        labels = np.ndarray.tolist(kmeans.labels_)
        clusterPoints = []
        for m in range(25):
            index = np.asarray([i for i, x in enumerate(labels) if x == m])
            clusterPoints.append(index)
        
        #finds beta values for gaussian by calculating distance to center
        betas = np.array([])
        for n in range(25):     #for each cluster
            dist = 0
            numPoints = len(clusterPoints[n])
            for p in range(numPoints):  #for each point in cluster
                dist += np.linalg.norm(training[clusterPoints[n][p]]-kmeans.cluster_centers_[n], ord=2)
            var = (dist/numPoints)**2
            beta = 1/(2*var)
            betas = np.append(betas,beta)   #stores beta values for each cluster
        betaVals.append(betas)
        
        #computes hidden layer neuron values for training images input
        gaussiansTrain = np.zeros(25)
        for image in training:
            gauss = np.array([])
            for n in range(25):
                if(drop[n]==1):
                    gauss = np.append(gauss,gaussian(image,clusterCenters[i][n],betaVals[i][n]))
                else:
                    gauss = np.append(gauss,0)   #dropout - set node to zero
            gaussiansTrain = np.vstack((gaussiansTrain,gauss))
        gaussiansTrain = gaussiansTrain[1:]
        
        #computes hidden layer neuron values for testing images input
        gaussiansTest = np.zeros(25)
        for image in test:
            gauss = np.array([])
            for n in range(25):
                gauss = np.append(gauss,gaussian(image,clusterCenters[i][n],betaVals[i][n]))
            gaussiansTest = np.vstack((gaussiansTest,gauss))
        gaussiansTest = gaussiansTest[1:]
        
        clf = MLPClassifier(solver = "lbfgs", alpha=1e-5, hidden_layer_sizes=(25,), activation = "identity")
        clf.fit(gaussiansTrain, trainLabels)
        predicted = clf.predict(gaussiansTest)
        
        correct = 0
        for t in range(size):
            if (np.array_equal(predicted[t], testLabels[t])):
                correct += 1
        
        percentage = 100*correct/size
        print("Percentage = ", percentage,"%")        
                
#        X = tf.placeholder("float", [None, 25])
#        Y = tf.placeholder("float", [None, 10])
#        w_0 = init_weights([25, 10]) 
#        
#        py_x = tf.matmul(X,w_0)
#        
#        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
#        train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
#        predict_op = tf.argmax(py_x, 1)
#        
#        saver = tf.train.Saver()
#        
#        # Launch the graph in a session
#        with tf.Session() as sess:
#            # you need to initialize all variables
#            tf.global_variables_initializer().run()
#            for start, end in zip(range(0, len(gaussiansTrain), 128), range(128, len(gaussiansTrain)+1, 128)):
#                sess.run(train_op, feed_dict={X: gaussiansTrain[start:end], Y: trainLabels[start:end]})
#            pred = sess.run(py_x, feed_dict={X: gaussiansTrain})
#            output = sess.run(predict_op, feed_dict={X: gaussiansTest})
#            print(np.mean(np.argmax(testLabels, axis=1) == output))
#            saver.save(sess,"mlp/session.ckpt")       


#plt.figure()
#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#plt.plot(kVals, dist)
#title = "K-Mean Cluster"
#plt.title(title)
#plt.xlabel("K")
#plt.ylabel("Distance")
#plt.grid()
#plt.show()

#### RBG Network
#def gaussAct(x):
#    b = 0.5
#    c = 
#    return tf.math.exp(-0.5*(x-))