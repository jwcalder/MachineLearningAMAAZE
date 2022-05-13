import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

def frag_to_break(x_frag,y_frag,num_breaks_per_frag):
    num_fragments = x_frag.shape[0]
    num_features = x_frag.shape[1]
    n = num_fragments*num_breaks_per_frag
    x = np.ones((num_fragments,num_breaks_per_frag,num_features))*x_frag[:,None,:]
    y = np.ones((num_fragments,num_breaks_per_frag))*y_frag
    x = np.reshape(x,(n,num_features))
    y = np.reshape(y,(n,))

    return x,y

def one_trial(num_fragments,num_breaks_per_frag,num_features):
    x_frag = np.random.rand(num_fragments,num_features)
    y_frag = (np.random.rand(num_fragments,1) > 0.5).astype(int)

    #Splitting at break level
    x,y = frag_to_break(x_frag,y_frag,num_breaks_per_frag)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    rf_pred = RandomForestClassifier().fit(x_train,y_train).predict(x_test)
    accuracy = accuracy_score(rf_pred, y_test)
    print('RF,break-level,%d,%d,%d,%f'%(num_fragments,num_breaks_per_frag,num_features,accuracy),flush=True)

    svm_pred = svm.SVC(kernel="linear").fit(x_train,y_train).predict(x_test)
    accuracy = accuracy_score(svm_pred, y_test)
    print('SVM,break-level,%d,%d,%d,%f'%(num_fragments,num_breaks_per_frag,num_features,accuracy),flush=True)

    #Splitting at fragment level
    x_frag_train, x_frag_test, y_frag_train, y_frag_test = train_test_split(x_frag, y_frag, test_size=0.25)
    x_train,y_train = frag_to_break(x_frag_train,y_frag_train,num_breaks_per_frag)
    x_test,y_test = frag_to_break(x_frag_test,y_frag_test,num_breaks_per_frag)

    rf_pred = RandomForestClassifier().fit(x_train,y_train).predict(x_test)
    accuracy = accuracy_score(rf_pred, y_test)
    print('RF,frag-level,%d,%d,%d,%f'%(num_fragments,num_breaks_per_frag,num_features,accuracy),flush=True)

    svm_pred = svm.SVC(kernel="linear").fit(x_train,y_train).predict(x_test)
    accuracy = accuracy_score(svm_pred, y_test)
    print('SVM,frag-level,%d,%d,%d,%f'%(num_fragments,num_breaks_per_frag,num_features,accuracy),flush=True)


print('Method,Split Type,Number of Fragments,Number of Breaks Per Fragment,Number of Features,Accuracy',flush=True)
for _ in range(10):
    for num_fragments in [100,1000]:
        for num_breaks_per_frag in [2,5,10,20]:
            for num_features in [10,20,50,100]:
                one_trial(num_fragments,num_breaks_per_frag,num_features)












