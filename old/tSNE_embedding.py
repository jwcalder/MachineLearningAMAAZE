from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import graphlearning as gl

from utils import frag_level_ml_dataset

data,target,specimens,target_names = frag_level_ml_dataset()

from sklearn import preprocessing
data = preprocessing.StandardScaler().fit_transform(data)
#Graph Settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 12})
styles = ['^-', 'o-', 'd-', 's-', 'p-', 'x-', '*-']

#T-SNE embedding, 2D
data_embedded = TSNE(n_components = 2, learning_rate = 'auto', init = 'random', perplexity = 5).fit_transform(data)

plt.scatter(data_embedded[:,0],data_embedded[:,1], c = target)
plt.title('TSNE, 2D')
plt.savefig('figures/tsne_2D')
plt.show()

#See KMeans accuracy on this graph
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(data_embedded)
kml = kmeans.labels_
print(gl.clustering.clustering_accuracy(kml, target))

#T-SNE embedding, 3D
data_embedded = TSNE(n_components = 3, learning_rate = 'auto', init = 'random', perplexity=5).fit_transform(data)

ax = plt.axes(projection ="3d")
ax.scatter3D(data_embedded[:,0],data_embedded[:,1],data_embedded[:,2],c=target,s=10)
ax.view_init(30, 60)
plt.title('TSNE, 3D')
plt.savefig('figures/tsne_3D')
plt.show()

#http://blog.mahler83.net/2019/10/rotating-3d-t-sne-animated-gif-scatterplot-with-matplotlib/
#Save animation as GIF

#See KMeans accuracy on this graph
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(data_embedded)
kml = kmeans.labels_
print(gl.clustering.clustering_accuracy(kml, target))

#Spectral clustering on this graph
from sklearn import preprocessing
data = preprocessing.StandardScaler().fit_transform(data)

W = gl.weightmatrix.knn(data_embedded,10)

avgAcc = 0
for i in range(100):
  model = gl.clustering.spectral(W,num_clusters = 2,method='NgJordanWeiss')

  pred_labels= model.fit_predict()
  accuracy = gl.clustering.clustering_accuracy(pred_labels,target)
  avgAcc += accuracy
print(avgAcc/100.0)