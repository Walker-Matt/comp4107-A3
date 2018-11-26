import numpy as np
import matplotlib.pyplot as plt
import os.path as path
from sklearn.datasets import fetch_mldata
from matplotlib.pyplot import figure
import sompy
from sklearn.cluster import KMeans

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
np.random.shuffle(training)    #shuffle order of images
    
#SOM clustering
mapsize = [28,28]
som = sompy.SOMFactory.build(training, mapsize, mask=None, mapshape='planar', 
                             lattice='rect', normalization='var', initialization='pca', 
                             neighborhood='gaussian', training='batch', name='sompy')
som.train(n_job=1, verbose='info')
v = sompy.mapview.View2DPacked(50, 50, 'test', text_size=8)
#v.show(som, what='codebook', cmap=None, col_sz=6)

cl = som.cluster(n_clusters=2)
SOMcluster = getattr(som, 'cluster_labels')
SOMcluster.shape = (28,28)
print(SOMcluster)

#SVD of matrix of images
U,s,V = np.linalg.svd(training, full_matrices = False)
s_inv = np.linalg.inv(np.diag(s))
US = np.matmul(U[0:784,0:2],s_inv[0:2,0:2])
training2D = np.matmul(training,US)

#computes kmeans to cluster images
kmeans = KMeans(n_clusters=2, random_state=0).fit(training)
labelsList = np.ndarray.tolist(kmeans.labels_)
clusterPoints = []

#identifies which points (images) are in which cluster
for k in range(2):
    index = np.asarray([i for i, x in enumerate(labelsList) if x == k])
    clusterPoints.append(index)

colors = np.zeros(len(training2D))
colors[clusterPoints[1]] = 1
training2D = training2D.T
training2D = 100*training2D

plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(training2D[0],training2D[1], c=colors)
title = "2D K-means Plot"
plt.title(title)
plt.xlabel("Dim1")
plt.ylabel("Dim2")
plt.grid()
plt.show()