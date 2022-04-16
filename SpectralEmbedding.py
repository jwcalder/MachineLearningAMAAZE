import graphlearning as gl
import matplotlib.pyplot as plt


from utils import frag_level_ml_dataset

data,target,specimens,target_names = frag_level_ml_dataset()

#Accuracy calculation for spectral clustering

from sklearn import preprocessing
data = preprocessing.StandardScaler().fit_transform(data)

W = gl.weightmatrix.knn(data,10)

avgAcc = 0
for i in range(100):
  model = gl.clustering.spectral(W,num_clusters = 2,method='NgJordanWeiss')

  pred_labels= model.fit_predict()
  accuracy = gl.clustering.clustering_accuracy(pred_labels,target)
  avgAcc += accuracy
print(avgAcc/100.0)

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

#3D graph
ax = plt.axes(projection ="3d")
ax.scatter3D(vec[:,1],vec[:,2],vec[:,3],c=target,s=10)
ax.view_init(30, 60)
plt.title('Spectral Embedding, 3D')
plt.savefig('figures/spectral_embedding_3D')
plt.show()
