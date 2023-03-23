import graphlearning as gl
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from utils import break_level_ml_dataset
from utils import frag_level_ml_dataset
from sklearn import preprocessing

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 12})
styles = ['^-', 'o-', 'd-', 's-', 'p-', 'x-', '*-']


def sc(data, tag, size, color):
    print(tag)
    W = gl.weightmatrix.knn(data, 20, similarity = 'angular')

    avgAcc = 0

    model = gl.clustering.spectral(W,num_clusters = 2,method='NgJordanWeiss')

    pred_labels= model.fit_predict()
    accuracy = gl.clustering.clustering_accuracy(pred_labels,target)
    if (accuracy_score(target,pred_labels) < accuracy_score(target,1-pred_labels)):
        pred_labels = 1 - pred_labels
    print('Total accuracy: %f' %accuracy)
    c = confusion_matrix(target,pred_labels)
    print('HSAnv accuracy: %f' %(100 * c[0,0]/(c[0,0] + c[0,1])))
    print('Teeth accuracy: %f' %(100 * c[1,1]/(c[1,1]+c[1,0])))

    print(c)
    # Creating graphs from spectral embedding
    G = gl.graph(W)
    print('Graph is connected:')
    print(G.isconnected())
    vals, vec = G.eigen_decomp(normalization='normalized')

    # 2D graph
    plt.figure()
    plt.scatter(vec[:, 1], vec[:, 2], c=target, vmax = 1.5, s = size, cmap = color)
    plt.savefig('figures/spectral_embedding_2D_' + tag + '.pdf')

    plt.figure()
    plt.scatter(vec[:, 1], vec[:, 2], c=pred_labels, vmax = 1.5, s = size, cmap = color)
    plt.savefig('figures/spectral_clustering_2D_' + tag + '.pdf')

    return vec

data,target,specimens,break_numbers,target_names = break_level_ml_dataset()
data = preprocessing.StandardScaler().fit_transform(data)
vec = sc(data, tag = 'break_level', size = 3, color = 'viridis')


data,target,specimens,target_names = frag_level_ml_dataset(numerical_fields=[], categorical_fields=[])
data = preprocessing.StandardScaler().fit_transform(data)
sc(data, tag = 'frag_level', size = 10, color = 'viridis')
plt.show()

