import numpy as np
import random
import matplotlib.pyplot as plt
import os.path as path
from sklearn.datasets import fetch_mldata
#
#Getting the raw data from the database
datafile = "mnist-original.mat"
if(not path.isfile(datafile)):
    try:
        print("Downloading MNIST data from mldata.org")
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
    plt.imshow(pixels, cmap='gray')
    plt.show()
    
#Convert mnist data into binary values (1 or -1)
def convert(data):
    binary = np.zeros((12162, 784), dtype=np.int)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if(data[i][j] > 0):
                binary[i][j] = 1
            else:
                binary[i][j] = -1
    return binary
    
#Creating the subsets
mnist = {"data": mnist_raw["data"].T,"labels": mnist_raw["label"][0]}
ones = mnist["data"][5923:12664]
fives = mnist["data"][30596:36017]
ones_labels = mnist["labels"][5923:12664]
fives_labels = mnist["labels"][30596:36017]

data = np.concatenate((ones, fives))
print("Converting mnist data to binary...")
data = convert(data)
labels = np.append(ones_labels, fives_labels)
    
#Hopfield network can only store about 0.15N patterns
#0.15 * number of neurons = 0.15 * 784 = 117.6 ~ 118
trainMax = 118
testMax = 1000
neuronNum = 784

steps = 10 #Reconstructs test data this many times

#Generate training and testing data/labels
def generate(num):
    indexes = random.sample(range(len(data)), num)
    return data[indexes[:trainMax]], labels[indexes[:trainMax]], data[indexes[trainMax:]], labels[indexes[trainMax:]]

trainX, trainY, testX, testY = generate(trainMax + testMax)

def train(neurons, training):
    w = np.zeros([neurons, neurons])
    for data in training:
        w += np.outer(data, data)
    for diagonal in range(neurons):
        w[diagonal][diagonal] = 0
    return w

def test(weights):
    output = []

    for image in testX:
        predicted = reconstruct(weights, image)
        output.append([image, predicted])
    
    return output

#Reconstruct data from weight matrix
def reconstruct(weights, data):
    res = np.array(data)

    for _ in range(steps):
        for i in range(len(res)):
            raw_v = np.dot(weights[i], res)
            if raw_v > 0:
                res[i] = 1
            else:
                res[i] = -1
    return res

# Train
print("Training the network...")
weights = train(neuronNum, trainX)

# Test
print("Testing the network...")
predictImgs = test(weights)

for i in range(10):
    display(predictImgs[i][0], "Test")
    display(predictImgs[i][1], "Predicted")
    