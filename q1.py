import numpy as np
import tensorflow as tf
import os.path as path
from sklearn.datasets import fetch_mldata

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
    
#Creating the subsets
mnist = {"data": mnist_raw["data"].T,"labels": mnist_raw["label"][0]}
ones = mnist["data"][5923:12664]
fives = mnist["data"][30596:36017]
ones_labels = mnist["labels"][5923:12664]
fives_labels = mnist["labels"][30596:36017]

trX = np.concatenate((ones, fives))
trY = np.append(ones_labels, fives_labels)

