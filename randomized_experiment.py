import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import Net

def frag_to_break(x_frag,y_frag,num_breaks_per_frag,num_break_features):
    """Fragment level to break level conveter
    ============

    Converts fragment level data (x_frag,y_frag) to break level data
    by copying to num_breaks_per_frag breaks. Also adds num_break_features 
    of random features to each break.
    
    Parameters
    ----------
    x_frag : numpy array (float)
        Fragment level features (num_fragments x num_features).
    y_frag : numpy array (int)
        Fragment level class labels.
    num_breaks_per_frag : int
        Number of breaks to create per fragment.
    num_break_features : int
        Number of additional break features to add (randomized).

    Returns
    -------
    x : numpy array (float)
        Break level features (num_breaks_per_frag * num_fragments) x (num_features + num_break_features).
    y : numpy array (int)
        Break level labels.
    """

    num_fragments = x_frag.shape[0]
    num_features = x_frag.shape[1]
    n = num_fragments*num_breaks_per_frag
    x = np.ones((num_fragments,num_breaks_per_frag,num_features))*x_frag[:,None,:]
    y = np.ones((num_fragments,num_breaks_per_frag))*y_frag[:,None]
    x = np.reshape(x,(n,num_features))
    y = np.reshape(y,(n,))

    if num_break_features > 0:
        x_break = np.random.rand(n,num_break_features)
        x = np.hstack((x,x_break))

    return x,y

num_fragments = 10
num_features = 20
num_breaks_per_frag = 7
num_break_features = 2

x_frag = np.random.rand(num_fragments,num_features)
y_frag = (np.random.rand(num_fragments) > 0.5).astype(int)

#Splitting at break level
x,y = frag_to_break(x_frag,y_frag,num_breaks_per_frag,num_break_features)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

models = []
models += [LinearDiscriminantAnalysis()]
models += [RandomForestClassifier()]
models += [svm.SVC(kernel="linear")]
models += [svm.SVC(kernel="rbf")]
models += [KNeighborsClassifier(n_neighbors=1)]
models += [Net(structure=[x.shape[1],100,200,400], num_classes=2)]

for model in models:
    print(model)
#pred = model.fit(x_train,y_train).predict(x_test)
#print({model})

#Splitting at fragment level
x_frag_train, x_frag_test, y_frag_train, y_frag_test = train_test_split(x_frag, y_frag, test_size=0.25)
x_train,y_train = frag_to_break(x_frag_train,y_frag_train,num_breaks_per_frag,num_break_features)
x_test,y_test = frag_to_break(x_frag_test,y_frag_test,num_breaks_per_frag,num_break_features)

#def one_trial(num_fragments,num_breaks_per_frag,num_features):
#    x_frag = np.random.rand(num_fragments,num_features)
#    y_frag = (np.random.rand(num_fragments,1) > 0.5).astype(int)
#
#    #Splitting at break level
#    x,y = frag_to_break(x_frag,y_frag,num_breaks_per_frag,2)
#    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
#    rf_pred = RandomForestClassifier().fit(x_train,y_train).predict(x_test)
#    accuracy = accuracy_score(rf_pred, y_test)
#    print('RF,break-level,%d,%d,%d,%f'%(num_fragments,num_breaks_per_frag,num_features,accuracy),flush=True)
#
#    svm_pred = svm.SVC(kernel="linear").fit(x_train,y_train).predict(x_test)
#    accuracy = accuracy_score(svm_pred, y_test)
#    print('SVM,break-level,%d,%d,%d,%f'%(num_fragments,num_breaks_per_frag,num_features,accuracy),flush=True)
#
#    #Splitting at fragment level
#    x_frag_train, x_frag_test, y_frag_train, y_frag_test = train_test_split(x_frag, y_frag, test_size=0.25)
#    x_train,y_train = frag_to_break(x_frag_train,y_frag_train,num_breaks_per_frag,2)
#    x_test,y_test = frag_to_break(x_frag_test,y_frag_test,num_breaks_per_frag,2)
#
#    rf_pred = RandomForestClassifier().fit(x_train,y_train).predict(x_test)
#    accuracy = accuracy_score(rf_pred, y_test)
#    print('RF,frag-level,%d,%d,%d,%f'%(num_fragments,num_breaks_per_frag,num_features,accuracy),flush=True)
#
#    svm_pred = svm.SVC(kernel="linear").fit(x_train,y_train).predict(x_test)
#    accuracy = accuracy_score(svm_pred, y_test)
#    print('SVM,frag-level,%d,%d,%d,%f'%(num_fragments,num_breaks_per_frag,num_features,accuracy),flush=True)
#
#
#print('Method,Split Type,Number of Fragments,Number of Breaks Per Fragment,Number of Features,Accuracy',flush=True)
#for _ in range(10):
#    for num_fragments in [100,1000]:
#        for num_breaks_per_frag in [2,5,10,20]:
#            for num_features in [10,20,50,100]:
#                one_trial(num_fragments,num_breaks_per_frag,num_features)












