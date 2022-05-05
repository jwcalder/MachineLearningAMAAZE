import graphlearning as gl
import matplotlib.pyplot as plt
import numpy as np
import sys

from utils import frag_level_ml_dataset

data,target,specimens,target_names = frag_level_ml_dataset()

#Accuracy calculation for spectral clustering

from sklearn import preprocessing
data = preprocessing.StandardScaler().fit_transform(data)

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

#K means Clustering
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2).fit_predict(data)
if (accuracy_score(target,km) < accuracy_score(target,1-km)):
    km = 1 - km

print(accuracy_score(target,km))
c = confusion_matrix(target,km)
print(c)

#LDA embedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#l = LDA(n_components = 1)
#data_lda = l.fit_transform(data, target)
#plt.scatter(data_lda[:,0], data_lda[:,1], c =  target)
#plt.show()

W = gl.weightmatrix.knn(data,10)

avgAcc = 0

model = gl.clustering.spectral(W,num_clusters = 2,method='NgJordanWeiss')

pred_labels= model.fit_predict()
accuracy = gl.clustering.clustering_accuracy(pred_labels,target)
if (accuracy_score(target,pred_labels) < accuracy_score(target,1-pred_labels)):
    pred_labels = 1 - pred_labels
print(accuracy)
c = confusion_matrix(target,pred_labels)
print(c)

#Graph Settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 12})
styles = ['^-', 'o-', 'd-', 's-', 'p-', 'x-', '*-']

#Creating graphs from spectral embedding
G = gl.graph(W)
vals,vec = G.eigen_decomp(normalization = 'normalized')

#2D graph
plt.scatter(vec[:,1], vec[:, 2], c = target)
plt.title('Spectral Embedding, 2D')
plt.savefig('figures/spectral_embedding_2D')
plt.show()

#Euclidean GBL
sys.stdout = open('results/gbl_spectralEmbed_ml_accuracies_euclidean.csv', 'w')
specEData = np.array([vec[:,1],vec[:,2]]).transpose()

W = gl.weightmatrix.knn(specEData,10)
x = 0
print('Percentage_Training_Data', end = ', ')
for i in range(100):
    print('Trial %d'%(i+1), end = ', ')
print('')
for i in range(15):
    avgAcc = 0
    x += 0.05
    x = round(x, 2)
    print(x, end = ', ')
    for j in range(100):
        train_ind = gl.trainsets.generate(target, rate=x)
        train_labels = target[train_ind]

        model = gl.ssl.laplace(W)
        pred_labels = model.fit_predict(train_ind, train_labels)

        accuracy = gl.ssl.ssl_accuracy(pred_labels, target, len(train_ind))
        print(accuracy, end = ', ')
    print('')
sys.stdout.close()

#3D graph
ax = plt.axes(projection ="3d")
ax.scatter3D(vec[:,1],vec[:,2],vec[:,3],c=target,s=10)
ax.view_init(30, 60)
plt.title('Spectral Embedding, 3D')
plt.savefig('figures/spectral_embedding_3D')
#plt.show()
#http://blog.mahler83.net/2019/10/rotating-3d-t-sne-animated-gif-scatterplot-with-matplotlib/
#Save animation as GIF