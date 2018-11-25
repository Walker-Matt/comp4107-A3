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
    
def gaussian(image,center,beta):
    return np.exp(-beta*(np.linalg.norm(image-center,ord=2)**2))

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
enc = OneHotEncoder(handle_unknown='ignore')
labels = enc.fit_transform(labels).toarray()

numImages = 5000
indices = random.sample(range(len(data)), numImages)
averages = np.array([])
#k-fold algorithm
kFold = np.array([6,9,12,15])
#kFold = np.array([5])
for k in kFold:
    percentages = np.array([])
    size = int(numImages/k)
    groups = np.zeros((k,size,784))
    labs = np.zeros((k,size,10))
    
    for i in range(k):
        groups[i] = data[indices[i*size:(i*size)+size]]
        labs[i] = labels[indices[i*size:(i*size)+size]]
    
    clusterCenters = []
    betaVals = []
    for i in range(k):
        print("Pass through: ", i+1)
        train = np.zeros((size,784))
        trainLabels = np.zeros((size,10))
        test = groups[i]
        testLabels = np.array(labs[i], dtype = int)
        for j in range(k):
            if (not i==j):
                train = np.vstack((train,groups[j]))
                trainLabels = np.vstack((trainLabels,labs[j]))
        training = train[size:]
        trainLabels = np.array(trainLabels[size:], dtype = int)
        kmeans = KMeans(n_clusters=25, random_state=0).fit(training)
        clusterCenters.append(kmeans.cluster_centers_)
#        clusters.append(np.array([kmeans.cluster_centers_, kmeans.labels_, kmeans.inertia_]))
        
        labelsList = np.ndarray.tolist(kmeans.labels_)
        clusterPoints = []
        for m in range(25):
            index = np.asarray([i for i, x in enumerate(labelsList) if x == m])
            clusterPoints.append(index)
        
        #finds beta values
        betas = np.array([])
        for n in range(25):
            dist = 0
            numPoints = len(clusterPoints[n])
            for p in range(numPoints):
                dist += np.linalg.norm(training[clusterPoints[n][p]]-kmeans.cluster_centers_[n], ord=2)
            beta = 1/(2*((dist/numPoints)**2))
            betas = np.append(betas,beta)
        betaVals.append(betas)
        
        gaussiansTrain = np.zeros(25)
        for image in training:
            gauss = np.array([])
            for n in range(25):
                gauss = np.append(gauss,gaussian(image,clusterCenters[i][n],betaVals[i][n]))
            gaussiansTrain = np.vstack((gaussiansTrain,gauss))
        gaussiansTrain = gaussiansTrain[1:]
        
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
        
        percent = round(100*correct/size,2)
        print("Percentage = ", percent, "%")
        
        percentages = np.append(percentages, percent)
        
    avg = np.mean(percentages)
    averages = np.append(averages,avg)

plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(kFold, averages)
title = "Accuracy vs. K-Fold"
plt.title(title)
plt.xlabel("K")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.show()