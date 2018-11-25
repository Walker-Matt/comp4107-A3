import numpy as np
import matplotlib.pyplot as plt
import os.path as path
from sklearn.datasets import fetch_mldata
from matplotlib.pyplot import figure
import sompy
import pandas as pd

#Getting the raw data from the database
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
    plt.imshow(pixels, cmap='gray')
    plt.show()

#Creating the subsets
if(mldata):
    mnist = {"data": mnist_raw["data"],"labels": mnist_raw["target"]}
else:
    mnist = {"data": mnist_raw["data"].T,"labels": mnist_raw["label"][0]}

#separate images of ones and fives
ones = mnist["data"][5923:12664]
fives = mnist["data"][30596:36017]
ones_labels = mnist["labels"][5923:12664]
fives_labels = mnist["labels"][30596:36017]

training = np.concatenate((ones,fives))

fig = plt.figure()
plt.plot(training[:,0],training[:,1],'ob',alpha=0.2, markersize=4)
fig.set_size_inches(7,7)

mapsize = [30,30]
som = sompy.SOMFactory.build(training, mapsize, mask=None, mapshape='planar', 
                             lattice='rect', normalization='var', initialization='pca', 
                             neighborhood='gaussian', training='batch', name='sompy')
som.train(n_job=1, verbose='info')
v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)
v.show(som, what='codebook', which_dim=[0,1], cmap=None, col_sz=6)

#print("Converting mnist data to binary...")
#ones = np.where(ones > 0, 1, -1)
#fives = np.where(fives > 0, 1, -1)

