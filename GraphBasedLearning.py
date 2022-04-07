import graphlearning as gl

from utils import frag_level_ml_dataset

data,target,specimens,target_names = frag_level_ml_dataset()

#GL Semisupervised, with ss
from sklearn import preprocessing
data = preprocessing.StandardScaler().fit_transform(data)

W = gl.weightmatrix.knn(data,10, similarity = 'angular')

#x is the percentage of training data
x = 0.75
train_ind = gl.trainsets.generate(target, rate=x)
train_labels = target[train_ind]


model = gl.ssl.laplace(W)
pred_labels = model.fit_predict(train_ind, train_labels)

accuracy = gl.ssl.ssl_accuracy(pred_labels, target, len(train_ind))
print("Accuracy: %.2f%%"%accuracy)