import graphlearning as gl
import matplotlib.pyplot as plt


from utils import frag_level_ml_dataset

data,target,specimens,target_names = frag_level_ml_dataset()

#Look at Spectral Embedding.

from sklearn import preprocessing
data = preprocessing.StandardScaler().fit_transform(data)

W = gl.weightmatrix.knn(data,10)

avgAcc = 0
for i in range(100):
  model = gl.clustering.spectral(W,num_clusters = 5,method='NgJordanWeiss')

  pred_labels= model.fit_predict()
  accuracy = gl.clustering.clustering_accuracy(pred_labels,target)
  avgAcc += accuracy
print(avgAcc/100.0)

G = gl.graph(W)
vals,vec = G.eigen_decomp(normalization = 'normalized')

plt.scatter(vec[:,1], vec[:, 2], c = target)
plt.savefig('figures/spectral_embedding')

plt.show()

#ax = plt.axes(projection ="3d")
#ax.scatter3D(vec[:,1],vec[:,2],vec[:,3],c=target,s=10)
#for angle in range(0, 360):
  #ax.view_init(30, angle)
  #plt.draw()
  #plt.pause(.001)