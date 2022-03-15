import numpy as np
import graphlearning as gl
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, RepeatedKFold
from sklearn import preprocessing
class GraphLearning:

    def __init__(self, W, k, algorithm):
        #X = features as (n,m) numpy array
        #k = number of neighbors for weight matrix
        #algorithm would be the algorithm name, e.g., 'laplace'
        #You can add other arguments as you like

        self.k = k
        self.algorithm = algorithm
        self.W = W
        self.all_pred_labels = None
    #Added the percentage of training data you want
    def fit(self, y_train, train_pctg):
        #x_train needs to have the indices of all the points. Easiest–make it the first/last column.
        #This is the function you should define
        #75% of the data is what I found to be the best at classifying the
        train_ind = gl.trainsets.generate(y_train, rate=train_pctg)
        train_labels = y_train[train_ind]
        #Get train indices from x_train, I think we'll have to include them as part of the x data
        self.all_pred_labels = gl.graph_ssl(self.W, train_ind, train_labels, algorithm=self.algorithm)

    def predict(self, x_test):
        test_ind = np.array(x_test) #Get test indices from x_test
        return self.all_pred_labels[test_ind]

    def get_params(self):
      return {'W':self.W, 'k':self.k, 'algorithm': self.algorithm}

    def getMatrix(X, k):
        #Look at variational neural network.
        scaler = preprocessing.StandardScaler().fit(X)  # Scaling data
        X_scaled = scaler.transform(X)
        W = gl.weightmatrix.knn(X_scaled,k)
        return W
