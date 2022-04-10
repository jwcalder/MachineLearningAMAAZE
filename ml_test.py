import utils
import csv
import numpy as np
from warnings import simplefilter
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tqdm import trange
import itertools

def create_dataset(dataset):
    """Create Dataset
    ========

    Creates the ML dataset as defined by test and utils.py
    
    Parameters
    ----------
    dataset : string
        Either "breaks" or "frags". Determines the dataset to be created
    Returns
    -------
    data : numpy array (float)
        Features.
    target : numpy array (int)
        Targets.
    specimen : numpy array (float)
        List of specimen names.
    """
    
    if(dataset == "breaks"):
        data, target, specimens, _, _ = utils.break_level_ml_dataset()
    elif(dataset == "frags"):
        data, target, specimens, _ = utils.frag_level_ml_dataset()
    else:
        raise ValueError("dataset must be 'breaks' or 'frags'") 
    
    return data, target, specimens


def run_test(data, target, specimens, test_name, results = {}, reps = 300):
    """Run Test
    ========

    Runs the given test the given number of replications. 
    
    Parameters
    ----------
    data : numpy array (float)
        Features.
    target : numpy array (int)
        Targets.
    specimen : numpy array (float)
        List of specimen names.
    test_name : string
        Name of the test.
    results : python dictionary (optional), default = {}
        Holds the results of previous tests.
    reps : int (optional), default = 300
        How replications you want to do

    Returns
    -------
    results : python dictionary
        Dictionary that contains all the results. 
    """
    
    ### Set-up
    # This suppresses linear svm convergence warnings.
    simplefilter(action='ignore', category=ConvergenceWarning)
    
    r = {"Test" : test_name, "reps" : reps, "Results" : {}}
    
    # Not the most elegant solution, but these lists will hold the accuracy for each rep
    rf_acc = np.empty(reps)
    svmL_acc = np.empty(reps)
    svmRBF_acc = np.empty(reps)
    nn_acc = np.empty(reps)
    lda_acc = np.empty(reps)
    gnb_acc = np.empty(reps)
    knn_acc = np.empty(reps)
    
    iter_description = "Test: " + test_name + ". Reps"
    for i in trange(reps, desc=iter_description):
        data_train, target_train, data_test, target_test, frag_test = utils.train_test_split(data, target, specimens)
        
        ### The Tests
        
        # Random Forest
        clfrf = RandomForestClassifier()
        clfrf.fit(data_train, target_train)
        rf_pred = clfrf.predict(data_test)
        acc = utils.specimen_voting(rf_pred, target_test, frag_test)
        rf_acc[i] = 100 * acc
        
        # SVM - Linear
        clfsvmL = SVC(kernel ="linear", max_iter = 5000)
        scalar = StandardScaler() # SVM works best with standard scaled data
        pipeline = Pipeline([('transformer', scalar), ('estimator', clfsvmL)]) # That's what this does, in a safe manner
        pipeline.fit(data_train, target_train)
        pipeline_pred = pipeline.predict(data_test)
        acc = utils.specimen_voting(pipeline_pred, target_test, frag_test)
        svmL_acc[i] = 100 * acc
        
        # SVM - RBF
        clfsvmRBF = SVC(kernel ="rbf", max_iter = 5000)
        scalar = StandardScaler() # SVM works best with standard scaled data
        pipeline = Pipeline([('transformer', scalar), ('estimator', clfsvmRBF)]) # That's what this does, in a safe manner
        pipeline.fit(data_train, target_train)
        pipeline_pred = pipeline.predict(data_test)
        acc = utils.specimen_voting(pipeline_pred, target_test, frag_test)
        svmRBF_acc[i] = 100 * acc
        
        # Neural Network
        num_features = data.shape[1]
        num_classes = np.max(target)+1
        nn = utils.Net(structure=[num_features,100,1000,5000], num_classes=num_classes,
                        dropout_rate=0.4,epochs=100,learning_rate=1,cuda=True)
        nn_scalar = preprocessing.StandardScaler().fit(data_train)  # Scaling data
        nn.fit(nn_scalar.transform(data_train), target_train)
        nn_pred = nn.predict(nn_scalar.transform(data_test))
        acc = utils.specimen_voting(nn_pred, target_test, frag_test)
        nn_acc[i] = 100 * acc
        
        # LDA
        clfLDA = LinearDiscriminantAnalysis()
        clfLDA.fit(data_train, target_train)
        lda_pred = clfLDA.predict(data_test)
        acc = utils.specimen_voting(lda_pred, target_test, frag_test)
        lda_acc[i] = 100 * acc
        
        # Naive Bayes
        clfgnb = GaussianNB()
        clfgnb.fit(data_train, target_train)
        gnb_pred = clfgnb.predict(data_test)
        acc = utils.specimen_voting(gnb_pred, target_test, frag_test)
        gnb_acc[i] = 100 * acc
        
        # KNN
        clfknn= KNeighborsClassifier(n_neighbors = 25)
        scalar = StandardScaler() # SVM works best with standard scaled data
        pipeline = Pipeline([('transformer', scalar), ('estimator', clfknn)]) # That's what this does, in a safe manner
        pipeline.fit(data_train, target_train)
        pipeline_pred = pipeline.predict(data_test)
        acc = utils.specimen_voting(pipeline_pred, target_test, frag_test)
        knn_acc[i] = 100 * acc

    r["Results"]["Random Forest"] = {"accuracy" : np.mean(rf_acc), "std" : np.std(rf_acc), "full data" : rf_acc}
    r["Results"]["SVM - Linear"] = {"accuracy" : np.mean(svmL_acc), "std" : np.std(svmL_acc), "full data" : svmL_acc}
    r["Results"]["SVM - RBF"] = {"accuracy" : np.mean(svmRBF_acc), "std" : np.std(svmRBF_acc), "full data" : svmRBF_acc}
    r["Results"]["Neural Network"] = {"accuracy" : np.mean(nn_acc), "std" : np.std(nn_acc), "full data" : nn_acc}
    r["Results"]["LDA"] = {"accuracy" : np.mean(lda_acc), "std" : np.std(lda_acc), "full data" : lda_acc}
    r["Results"]["Naive Bayes"] = {"accuracy" : np.mean(gnb_acc), "std" : np.std(gnb_acc), "full data" : gnb_acc}
    r["Results"]["KNN"] = {"accuracy" : np.mean(knn_acc), "std" : np.std(knn_acc), "full data" : knn_acc}
    
    results[test_name] = r
    
    return results


def save_results(results, fname = "AMAAZE_test_results.csv", full = False, full_header = "AMAAZE_full_results_"):
    """Save Results
    ========

    Saves the results from the tests in a csv file. 
    
    Parameters
    ----------
    results : python dictionary
        Dictionary that contains all the results.
    fname : string
        Name of the file that the summary results are saved to.
    full : boolean (optional), default = False
        Whether the full results should be output for each replication
    full_header : string  (optional), default = "Full_results_"
        The header to each of the test results file.
    """
    
    # Summary Stats
    
    with open(fname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for test in results.keys():
            test_name = results[test]["Test"]
            reps = results[test]["reps"]
            writer.writerow(["Test: " + test_name, "Reps:", reps])
            writer.writerow(["Algorithm", "Mean Accuracy", "Standard Deviation"])
            for algo in results[test]["Results"].keys():
                name = algo
                acc = results[test]["Results"][algo]["accuracy"]
                std = results[test]["Results"][algo]["std"]
                writer.writerow([name, acc, std])
            writer.writerow([])
            
            
    # Full data
    
    if(full == True):
        for test in results.keys():
            test_name = results[test]["Test"]
            reps = results[test]["reps"]
            full_save_name = full_header + test_name + ".csv"
            with open(full_save_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Test: " + test_name, "Reps:", reps])
                writer.writerow(results[test]["Results"].keys()) # Column for each algorithm
                data = []
                for algo in results[test]["Results"].keys():
                    data.append(results[test]["Results"][algo]["full data"])
                rows = zip(*data)
                writer.writerows(rows)
        
def main():
    tests = ["breaks", "frags"]
    results = {}
    for test in tests:
        data, target, specimens = create_dataset(test)
        results = run_test(data, target, specimens, test)
    
    save_results(results, full = False)


if __name__ == '__main__':
    main()

