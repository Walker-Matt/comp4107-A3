import numpy as np
import matplotlib.pyplot as plt
import os.path as path
from sklearn.datasets import fetch_mldata
from matplotlib.pyplot import figure

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

print("Converting mnist data to binary...")
ones = np.where(ones > 0, 1, -1)
fives = np.where(fives > 0, 1, -1)

# similar to algorithm from stack exchange written by Russel Richie
#https://stats.stackexchange.com/questions/276889/whats-wrong-with-my-algorithm-for-implementing-the-storkey-learning-rule-for-ho
def train(neurons, training):
    w = np.zeros([neurons, neurons])
    for image in training:
        h = np.outer(image,image) - np.identity(neurons)
        t1 = w.dot(image)
        t2 = np.outer(image,t1)
        t3 = t2.T
        w = w + (1/neurons)*(h - t2 - t3)
    return w

#Reconstruct data from weight matrix
def reconstruct(weights, data):
    prev = np.zeros((neuronNum))
    res = np.array(data)
    steps = 0
    while(not np.array_equal(prev,res) and steps < 10): #Runs until stable
        prev = np.array(res)
        active = np.dot(weights, res)
        res = np.where(active > 0, 1, -1)
        steps += 1
    return res

# Hand-selected Ideal training images
training_ones = [99, 72, 96, 90, 21, 82, 54, 31, 47, 86]
training_fives = [99, 94, 93, 92, 89, 86, 83, 79, 74, 70]

testNum = 500
maxTrain = 21
neuronNum = 784

testOnes = np.array([], dtype=int)
testFives = np.array([], dtype=int)
for i in range(testNum):
    rand = np.random.choice([0,1])
    if (rand == 0):
        one_pos = np.random.randint(0, len(ones))
        while(np.isin(one_pos,training_ones)): #ensures test image isn't also a training image
           one_pos = np.random.randint(0, len(ones))
        testOnes = np.append(testOnes, one_pos)
    else:
        five_pos = np.random.randint(0, len(fives))
        while(np.isin(five_pos,training_fives)): #ensures test image isn't also a training image
            five_pos = np.random.randint(0, len(fives))
        testFives = np.append(testFives, five_pos)

percentages = np.array([])

for trainNum in range(1,maxTrain):       
    #Train
    print("Training the network with", trainNum, "images...")
    training = []
    prev = 0
    for i in range(trainNum):
        if(i % 2 == 0):
            training.append(ones[training_ones[i-prev]])
        else:
            training.append(fives[training_fives[i-prev]])
        if(i == 0):
            prev = 1
        else:
            prev = i
    #weight assignment using training images
    weights = train(neuronNum, training)
    
    correct = 0
    for i in range(len(testOnes)):
        predictedOne = reconstruct(weights, ones[testOnes[i]])
        normValsOnes = np.array([])
        for t in range(len(training)):
            norm = np.linalg.norm(training[t]-predictedOne, ord=2)
            normValsOnes = np.append(normValsOnes,norm)
        minPos = np.argmin(normValsOnes)
        if(minPos%2==0):
            correct += 1
    
    for i in range(len(testFives)):
        predictedFive = reconstruct(weights, fives[testFives[i]])
        normValsFives = np.array([])
        for t in range(len(training)):
            norm = np.linalg.norm(training[t]-predictedFive, ord=2)
            normValsFives = np.append(normValsFives,norm)
            minPos = np.argmin(normValsFives)
        if(minPos%2==1):
            correct += 1
            
    accuracy = 100*correct/(testNum)
    percentages = np.append(percentages,accuracy)
    
xlabel = np.arange(1,maxTrain)
plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(xlabel, percentages)
title = "Accuracy vs Number of Training Images"
plt.title(title)
plt.xlabel("Number of Training Images")
plt.ylabel("Accuracy (%)")
plt.xticks(xlabel)
plt.grid()
plt.show()