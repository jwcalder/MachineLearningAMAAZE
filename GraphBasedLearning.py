import graphlearning as gl

from utils import frag_level_ml_dataset
import sys

data,target,specimens,target_names = frag_level_ml_dataset()


#GL Semisupervised, with ss
from sklearn import preprocessing
data = preprocessing.StandardScaler().fit_transform(data)

W = gl.weightmatrix.knn(data,10, similarity = 'angular')
def getaccuracyGL(x):
    train_ind = gl.trainsets.generate(target, rate=x)
    train_labels = target[train_ind]


    model = gl.ssl.laplace(W)
    pred_labels = model.fit_predict(train_ind, train_labels)

    accuracy = gl.ssl.ssl_accuracy(pred_labels, target, len(train_ind))
    return accuracy
#We will list percentages of training data from 5% to 75%. We will run each percentage 500 times, and get the average of those 500 trials.
sys.stdout = open('results/gbl_ml_accuracies_angular.csv', 'w')
x = 0
print('Percentage_Training_Data', end = ', ')
for i in range(500):
    print('Trial %d'%(i+1), end = ', ')
print('Average_Accuracy')
for i in range(15):
    avgAcc = 0
    x += 0.05
    x = round(x, 2)
    print(x, end = ', ')
    for i in range(500):
        acc = getaccuracyGL(x)
        avgAcc += acc
        print(acc, end = ', ')
    print(avgAcc/500.0, end = '')
    print('')
sys.stdout.close()
sys.stdout = open('results/gbl_ml_accuracies_euclidean.csv', 'w')
W = gl.weightmatrix.knn(data,10)
x = 0
print('Percentage_Training_Data', end = ', ')
for i in range(500):
    print('Trial %d'%(i+1), end = ', ')
print('Average_Accuracy')
for i in range(15):
    avgAcc = 0
    x += 0.05
    x = round(x, 2)
    print(x, end = ', ')
    for i in range(500):
        acc = getaccuracyGL(x)
        avgAcc += acc
        print(acc, end = ', ')
    print(avgAcc/500.0, end = '')
    print('')
sys.stdout.close()



