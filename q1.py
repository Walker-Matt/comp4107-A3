import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os.path as path
from sklearn.datasets import fetch_mldata
#
#Getting the raw data from the database
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
ones = mnist["data"][5923:12664]
fives = mnist["data"][30596:36017]
ones_labels = mnist["labels"][5923:12664]
fives_labels = mnist["labels"][30596:36017]

trX = np.concatenate((ones, fives))
trY = np.append(ones_labels, fives_labels)
    
#Hopfield network can only store about 0.15N patterns
#0.15 * number of neurons = 0.15 * 784 = 117.6 ~ 118
trainNum = 118
testNum = len(trX) - 118
neuronNum = 784
