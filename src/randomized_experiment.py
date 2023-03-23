import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import Net
import pickle
from tqdm import tqdm

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

def train_test_models(models,x_train,y_train,x_test,y_test):
    """Train and test a list of models
    ============

    Trains and computes test accuracy over a list of models.
    
    Parameters
    ----------
    models : Python list
        List of machine learning models to test.
    x_train : numpy array (float)
        Training data.
    y_train : numpy array (int)
        Training labels
    x_test : numpy array (float)
        Testing data.
    y_test : numpy array (int)
        Testing labels
    
    Returns
    -------
    r : Python dictionary
        Python dictionary containing accuracies for all methods.
    """

    r = {}
    for model in models:
        model_name = str(model)
        model.fit(x_train,y_train)
        pred = model.predict(x_test)
        acc = 100*accuracy_score(y_test,pred)
        r[model_name] = acc
    return r

def append(d,r):
    for key in d:
        d[key] += [r[key]]

#Parameters for test
num_fragments = 200
num_features = 34
num_breaks_per_frag = 7
num_break_features = 6
bootstrap_factor = 100
num_trials = 100


#List of models
models = []
models += [LinearDiscriminantAnalysis()]
models += [RandomForestClassifier()]
models += [svm.SVC(kernel="linear")]
models += [svm.SVC(kernel="rbf")]
models += [KNeighborsClassifier(n_neighbors=1)]
models += [Net(structure=[num_features+num_break_features,100,200,400], num_classes=2, cuda=True)]

#Dictionaries for results
break_acc = {}
frag_acc = {}
boot_acc = {}
for model in models:
    model_name = str(model)
    break_acc[model_name] = []
    frag_acc[model_name] = []
    boot_acc[model_name] = []

for _ in tqdm(range(num_trials)):
    #Random fragment level data
    x_frag = np.random.rand(num_fragments,num_features)
    y_frag = (np.random.rand(num_fragments) > 0.5).astype(int)

    #Splitting at break level
    x,y = frag_to_break(x_frag,y_frag,num_breaks_per_frag,num_break_features)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    r = train_test_models(models,x_train,y_train,x_test,y_test)
    append(break_acc,r)

    #Bootstrapping
    ind = np.random.choice(num_fragments,size=bootstrap_factor*num_fragments)
    x,y = x[ind,:],y[ind]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    r = train_test_models(models,x_train,y_train,x_test,y_test)
    append(boot_acc,r)

    #Splitting at fragment level
    x_frag_train, x_frag_test, y_frag_train, y_frag_test = train_test_split(x_frag, y_frag, test_size=0.25)
    x_train,y_train = frag_to_break(x_frag_train,y_frag_train,num_breaks_per_frag,num_break_features)
    x_test,y_test = frag_to_break(x_frag_test,y_frag_test,num_breaks_per_frag,num_break_features)
    r = train_test_models(models,x_train,y_train,x_test,y_test)
    append(frag_acc,r)


#Write dictionaries of results to files
with open('../results/randomized_break.pkl','wb') as f:
    pickle.dump(break_acc,f)
with open('../results/randomized_frag.pkl','wb') as f:
    pickle.dump(frag_acc,f)
with open('../results/randomized_boot.pkl','wb') as f:
    pickle.dump(boot_acc,f)








