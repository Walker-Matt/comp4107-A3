import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

# split into a training and testing set using K-fold
Kfolds = [3,6,9,12,15]
percentages = []
for k in Kfolds:
    kf = KFold(n_splits=k)
    accuracies = []
    print("Testing with", k, "folds")
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        n_components = 150
        
        pca = PCA(n_components=n_components, svd_solver='randomized',
                  whiten=True).fit(X_train)
        
        eigenfaces = pca.components_.reshape((n_components, h, w))
        
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(n_components), random_state=1)
        clf = clf.fit(X_train_pca, y_train)
        
        y_pred = clf.predict(X_test_pca)
        
        def diff(test, pred):
            count = 0
            for i in range(len(test)):
                if(test[i] == pred[i]):
                    count += 1
            return count/len(test)
        
        accuracy = diff(y_test, y_pred)
        accuracies.append(accuracy)
    average_percent = int(sum(accuracies)/k * 100)
    percentages.append(average_percent)
    
plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(Kfolds,percentages, label = "Measured Data")
plt.title("K-Folds vs Network Accuracy")
plt.xlabel("K-Folds")
plt.ylabel("Recognition Percentage (%)")
plt.grid()
plt.show()