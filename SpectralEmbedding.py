import graphlearning as gl
import matplotlib.pyplot as plt


from utils import frag_level_ml_dataset

data,target,specimens,target_names = frag_level_ml_dataset()

#Look at Spectral Embedding.

from sklearn import preprocessing
data = preprocessing.StandardScaler().fit_transform(data)

W = gl.weightmatrix.knn(data,10)

vec = gl.spectral_embedding(W,4,method='NgJordanWeiss')
#vec is an nx2 dimensional array.
#Can plot vec with matplotlib.pyplot as plt.
ax = plt.gca()
ax.set_xlim([0.04,0.05])
ax.set_ylim([0,100])
print(vec[:, 1:])
plt.ion()
plt.scatter(vec[:,1], vec[:, 2], c = target)
plt.savefig('/Users/EricChen/Documents/Python/realdataplot.jpg')
plt.figure()

ax = plt.axes(projection ="3d")
ax.scatter3D(vec[:,1],vec[:,2],vec[:,3],c=target,s=10)
for angle in range(0, 360):
  ax.view_init(30, angle)
  plt.draw()
  plt.pause(.001)