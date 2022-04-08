import numpy as np
from utils import frag_level_ml_dataset, Net
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score

data,target,specimens,target_names = frag_level_ml_dataset()

num_features = data.shape[1]
num_classes = np.max(target)+1

model = Net(structure=[num_features,100,1000,5000], num_classes=num_classes,
            batch_normalization=False, dropout_rate=0.4)

T = 100
avg_acc = 0
for i in range(1,T+1):
    print('Trial #%d'%i)

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.25)

    #Standard scaler transform
    scaler = preprocessing.StandardScaler().fit(data_train)  # Scaling data
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    model.fit(data_train, target_train, epochs=100, batch_size=32, learning_rate=1)

    pred = model.predict(data_test)
    acc = accuracy_score(pred,target_test)
    print('Testing Accuracy: %.2f'%(100*acc))
    avg_acc += acc

    pred = model.predict(data_train)
    acc = accuracy_score(pred,target_train)
    print('Training Accuracy: %.2f'%(100*acc))

avg_acc /= T
print('Average Testing Accuracy: %.2f'%(100*avg_acc))


