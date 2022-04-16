from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from utils import frag_level_ml_dataset

data,target,specimens,target_names = frag_level_ml_dataset()

#T-SNE embedding, 2D
data_embedded = TSNE(n_components = 2, learning_rate = 'auto', init = 'random').fit_transform(data)

plt.scatter(data_embedded[:,0],data_embedded[:,1], c = target)
plt.savefig('figures/tsne_2D')
plt.show()


#T-SNE embedding, 3D
data_embedded = TSNE(n_components = 3, learning_rate = 'auto', init = 'random').fit_transform(data)

ax = plt.axes(projection ="3d")
ax.scatter3D(data_embedded[:,0],data_embedded[:,1],data_embedded[:,2],c=target,s=10)
ax.view_init(30, 60)
plt.savefig('figures/tsne_3D')
plt.show()
