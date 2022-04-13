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
#Angular GBL
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
    for j in range(500):
        acc = getaccuracyGL(x)
        avgAcc += acc
        print(acc, end = ', ')
    print(avgAcc/500.0, end = '')
    print('')
sys.stdout.close()
#Euclidean GBL
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
    for j in range(500):
        acc = getaccuracyGL(x)
        avgAcc += acc
        print(acc, end = ', ')
    print(avgAcc/500.0, end = '')
    print('')
sys.stdout.close()
#Random Forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
sys.stdout = open('results/rf_ml_accuracies.csv', 'w')
rf = RandomForestClassifier(n_estimators = 100)
x = 0
print('Percentage_Training_Data', end=', ')
print('Average_Accuracy', end = '')
print('')
for i in range (15):
    x += 0.05
    x = round(x,2)
    print(x, end = ', ')
    avgAcc = 0
    for j in range(50):
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=1 - x)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        avgAcc += metrics.accuracy_score(y_test, y_pred)
    avgAcc/= 50.0
    print(avgAcc)
sys.stdout.close()
