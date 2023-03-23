import csv
import pandas as pd
import numpy as np
from warnings import simplefilter
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from utils import Net

def moclan_dataset(bootstrap=False, bootstrap_num=1, level = 'moclan', dataset='moclan'):
    """Moclan Dataset
    ========

    Preps the moclan dataset for certain replication tests.
    
    Parameters
    ----------
    boostrap : boolean (optional), default = 'False'
        Determines whether a bootstrap is to be used.
    bootstrap_num : integer (optional), default = 1
        Determines how many additional bootstrap samples are to be added.
    level : string (optional), default = 'moclan'
        Selects the level of input features to include in data. 
        "moclan" includes all of the features.
        "breaks" includes only the break level features of angle and feature_plane
    dataset : string(optional", default = 'moclan')
        Selects the dataset to be used.
        "moclan" has all three effector levels, which is heavily unbalanced
        "carngrouped" has two effector levels, humans vs. carnivores

    Returns
    -------
    data : numpy array (float)
        Features.
    target : numpy array (int)
        Targets.
    test_name : string
        Name of the test.
    """
    
    test_name = level
    df = None
    if(dataset == "moclan"):
        df = pd.read_csv('../data/moclan.csv')
    elif(dataset == "carngrouped"):
        df = pd.read_csv('../data/moclan_carngrouped.csv')
        test_name = test_name + "_" + dataset
    else:
        raise ValueError("dataset must be 'moclan' or 'carngrouped'")
    
    if(bootstrap):
        test_name = test_name + "_bootstrap_" + str(bootstrap_num)
        boot_strap = df.sample(bootstrap_num, replace=True)
        df = pd.concat([df, boot_strap])
    
    le = preprocessing.LabelEncoder()
    target = le.fit_transform(df['Agent'])
    data = None
    
    if(level == "moclan"):
        num_data = df[['Length(mm)','Angle']].values
        cat_data = df[['Epiphysis', 'Number_of_planes', 'Type_of_angle', 
                       'Notch', 'Notch_a', 'Notch_c', 'Notch_d', 
                       'Fracture_plane', '>4cm', 
                       'Interval(length)']] 
        
        le = preprocessing.OneHotEncoder(sparse=False)
        cat_data = le.fit_transform(cat_data)
        
        data = np.hstack((cat_data, num_data))
    elif(level == "breaks"):
        num_data = df[['Angle']].values 
        cat_data = df[['Fracture_plane']]
        
        le = preprocessing.OneHotEncoder(sparse=False)
        cat_data = le.fit_transform(cat_data)
        
        data = np.hstack((cat_data, num_data))
    else:
        raise ValueError("level must be 'moclan' or 'breaks'")
        
    return (data, target, test_name)


def run_test(data, target, test_name, rf_np = False, reps = 300, folds = 10, k = 25, n_jobs=1):
    """Moclan Dataset
    ========

    Runs replication tests on the given dataset.
    
    Parameters
    ----------
    data : numpy array (float)
        Features.
    target : numpy array (int)
        Targets.
    test_name : string
        Name of the test.
    rf_np : boolean (optional), default = False
        If you want to run an additional bootstrapless random forest model
    reps : int (optional), default = 300
        How many k-fold replications you want to do.
    folds : int, default=10
        How many folds to use in k-fold replications.
    k : int (optional), default=25
        The number of k-nearest neighbors used in k-nearest neighbors classification
    n_jobs : int (optional), default=1
        Number of jobs to run in parallel.

    Returns
    -------
    r : python dictionary
        dictionary that contains results of current test.
    """
    
    ### Set-up
    # This suppresses linear svm convergence warnings.
    simplefilter(action='ignore', category=ConvergenceWarning)
    # This is our repeated-K-Fold object that ensures each test runs the same dataset
    cv = RepeatedKFold(n_splits = folds, n_repeats = reps)
    r = {"Test" : test_name, "reps" : reps, "folds" : folds, "Results" : {}}
    
    ### The Tests
    # Random Forests
    rf = RandomForestClassifier()
    rf_acc = 100*cross_val_score(rf, data, target, cv = cv, scoring = 'accuracy', n_jobs=n_jobs)
    r["Results"]["Random Forest"] = {"accuracy" : np.mean(rf_acc), "std" : np.std(rf_acc)}
    
    # Support Vector Machines
    clfsvm = SVC(kernel ="linear", max_iter = 5000)
    scalar = StandardScaler() # SVM works best with standard scaled data
    pipeline = Pipeline([('transformer', scalar), ('estimator', clfsvm)]) # That's what this does, in a safe manner
    svm_acc = 100*cross_val_score(pipeline, data, target, cv = cv, scoring = 'accuracy', n_jobs=n_jobs)
    r["Results"]["Linear SVM"] = {"accuracy" : np.mean(svm_acc), "std" : np.std(svm_acc)}

    # LDA
    clfLDA = LinearDiscriminantAnalysis()
    lda_acc = 100*cross_val_score(clfLDA, data, target, cv = cv, scoring='accuracy', n_jobs=n_jobs)
    r["Results"]["LDA"] = {"accuracy" : np.mean(lda_acc), "std" : np.std(lda_acc)}

    # Naive Bayes
    clfgnb = GaussianNB()
    gnb_acc = 100*cross_val_score(clfgnb, data, target, cv = cv, scoring='accuracy', n_jobs=n_jobs)
    r["Results"]["GNB"] = {"accuracy" : np.mean(gnb_acc), "std" : np.std(gnb_acc)}

    # KNN
    clfknn= KNeighborsClassifier(n_neighbors = k)
    scalar = StandardScaler()
    pipeline = Pipeline([('transformer', scalar), ('estimator', clfknn)])
    knn_acc = 100*cross_val_score(pipeline, data, target, cv = cv, scoring='accuracy', n_jobs=n_jobs)
    r["Results"]["KNN"] = {"accuracy" : np.mean(knn_acc), "std" : np.std(knn_acc)}

    #Neural Networks
    num_features = data.shape[1]
    num_classes = np.max(target)+1
    clfNet = Net(structure=[num_features,100,200,400], num_classes=num_classes)
    scalar = StandardScaler()
    pipeline = Pipeline([('transformer', scalar), ('estimator', clfNet)])
    Net_acc = 100*cross_val_score(pipeline, data, target, cv = cv, scoring='accuracy', n_jobs=n_jobs)
    r["Results"]["NeuralNet"] = {"accuracy" : np.mean(Net_acc), "std" : np.std(Net_acc)}
    
    # Random Forest no bootstrap
    if(rf_np):
        rf_np = RandomForestClassifier(bootstrap = False)
        rf_np_acc = 100*cross_val_score(rf, data, target, cv = cv, scoring = 'accuracy', n_jobs=n_jobs)
        r["Results"]["Random Forest (no bootstrap)"] = {"accuracy" : np.mean(rf_np_acc), "std" : np.std(rf_np_acc)}
    
    return r

def save_results(results, fname = "Moclan_Replications.csv"):
    """Moclan Dataset
    ========

    Saves the results from the tests in a csv file. 
    
    Parameters
    ----------
    results : python dictionary
        Dictionary that contains all the results.
    fname : string
        Name of the file that the results are saved to

    """
    fname = "../results/"+fname
    with open(fname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for test in results.keys():
            test_name = results[test]["Test"]
            reps = results[test]["reps"]
            folds = results[test]["folds"]
            writer.writerow(["Test: " + test_name, "Reps: " + str(reps), "Folds: " + str(folds)])
            writer.writerow(["Algorithm", "Mean Accuracy", "Standard Deviation"])
            for algo in results[test]["Results"].keys():
                name = algo
                acc = results[test]["Results"][algo]["accuracy"]
                std = results[test]["Results"][algo]["std"]
                writer.writerow([name, acc, std])
            writer.writerow([])
    
def main():
    
    """Main
    ========

    Runs the stated battery of replication tests. 
    
    """
    tests = ["moclan", "moclan", "breaks", "breaks"]
    dataset = ["moclan", "moclan", "moclan", "carngrouped"]
    bootstrap = [True, False, False, False]
    bootstrap_rep = [1000, 0, 0, 0]
    k = [1, 25, 25, 25] # k = 1 works best when bootstrapping
    results = {}
    
    for i in range(len(tests)):
        data, target, name = moclan_dataset(level=tests[i], dataset = dataset[i] , bootstrap = bootstrap[i], bootstrap_num = bootstrap_rep[i])
        results[name] = run_test(data, target, test_name = name, k = k[i], n_jobs=24)
    
    save_results(results)


if __name__ == '__main__':
    main()


