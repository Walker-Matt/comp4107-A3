import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape

X = lfw_people.data
n_features = X.shape[1]

y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

def diff(test, pred):
    count = 0
    for i in range(len(test)):
        if(test[i] == pred[i]):
            count += 1
    return count/len(test)

Kfolds = [3,6,9,12,15]
PCA_percentages = []
RAW_percentages = []
for k in Kfolds:
    # split into a training and testing set using K-fold
    kf = KFold(n_splits=k)
    PCA_accuracies = []
    RAW_accuracies = []
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
        
        #Feed forward neural network with back propagation
        clf_pca = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(n_components), random_state=1)
        clf_pca = clf_pca.fit(X_train_pca, y_train)
        
        y_pred_pca = clf_pca.predict(X_test_pca)
        
        PCA_accuracy = diff(y_test, y_pred_pca)
        PCA_accuracies.append(PCA_accuracy)
        
        #Testing with raw data
        clf_raw = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
        clf_raw = clf_pca.fit(X_train, y_train)
        
        y_pred_raw = clf_pca.predict(X_test)
        
        RAW_accuracy = diff(y_test, y_pred_raw)
        RAW_accuracies.append(RAW_accuracy)
        
    PCA_average_percent = int(sum(PCA_accuracies)/k * 100)
    PCA_percentages.append(PCA_average_percent)
    
    RAW_average_percent = int(sum(RAW_accuracies)/k * 100)
    RAW_percentages.append(RAW_average_percent)
    
plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(Kfolds,PCA_percentages, label = "Orthonormal Basis")
plt.plot(Kfolds,RAW_percentages, label = "Raw Data")
ax = plt.subplot(1,1,1)
ax.legend()
plt.title("Orthonormal Basis vs Raw Data")
plt.xlabel("K-Folds")
plt.ylabel("Recognition Percentage (%)")
plt.grid()
plt.show()