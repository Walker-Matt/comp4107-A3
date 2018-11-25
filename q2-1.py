from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
from sklearn.datasets import fetch_mldata
import random
from matplotlib.pyplot import figure

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
        mnist_raw = fetch_mldata('MNIST original')
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
    
#Creating the subsets
if(mldata):
    mnist = {"data": mnist_raw["data"],"labels": mnist_raw["target"]}
else:
    mnist = {"data": mnist_raw["data"].T,"labels": mnist_raw["label"][0]}
data = mnist['data']
labels = mnist['labels']

numImages = 1000    #number of images to use
indices = random.sample(range(len(data)), numImages)    #random sample of images
dist = np.array([])

c = 16000000    #adjusted c constant to determine optimized k value for kmeans
kVals = np.arange(10,40,1)  #array of k values from 10 to 40
for k in kVals:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data[indices])    #calculates kmeans using training images
    dist = np.append(dist,kmeans.inertia_ + c*k)    #total distance of points to nearest cluster center

plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(kVals, dist)
title = "K-Mean Cluster"
plt.title(title)
plt.xlabel("K")
plt.ylabel("Distance")
plt.grid()
plt.show()