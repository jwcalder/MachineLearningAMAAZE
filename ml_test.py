import utils
import csv
import numpy as np
import os
import sys
from pandas import DataFrame
from datetime import datetime
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from tqdm import trange

def create_dataset(options):
    """Create Dataset
    ========

    Creates the ML dataset as defined by test and utils.py

    Options Dictionary
    ----------
    dataset : string
        Either "breaks" or "frags". Determines the level of dataset to be created
    target_field : string
        Field to use for target of machine learning.
    name : string
        The name appended to the saved files
    numerical_fields : list or None
        List of numerical fields to use. Uses all if not provided.
    categorical_fields : list or None
        List of categorical fields to use. Uses all if not provided.
    sum_stats_fields : list or None
        List of summary statistics fields to use. Uses all if not provided.
    sum_stats : list or None
        Summary statistics to use. Uses all if not provided.
    count_fields : list or None
        List of categorical counting fields.
        
    Parameters
    ----------
    options : dictionary
        Contains all the options for the spesific test. dataset and target_field must be filled in. Defaultdict is recommended, so you don't have to define the optional fields you don't need. 
    
    Returns
    -------
    data : numpy array (float)
        Features.
    target : numpy array (int)
        Targets.
    specimen : numpy array (float)
        List of specimen names.
    name : string
        name appended to save files
    """
    
    dataset = options["dataset"]
    target_field = options["target_field"]
    numerical_fields = options["numerical_fields"] 
    categorical_fields = options["categorical_fields"] 
    sum_stats_fields = options["sum_stats_fields"] 
    sum_stats = options["sum_stats"] 
    count_fields = options["count_fields"] 
        
    if(dataset == "breaks"):
        data, target, specimens, _, _ = utils.break_level_ml_dataset(numerical_fields, categorical_fields, target_field)
    elif(dataset == "frags"):
        data, target, specimens, _ = utils.frag_level_ml_dataset(numerical_fields, categorical_fields, sum_stats_fields, sum_stats, count_fields, target_field)
    else:
        raise ValueError("dataset must be 'breaks' or 'frags'")

    return data, target, specimens

def run_test(data, target, specimens, dataset_level, desc, results={}, reps=3):
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
    dataset_level : string
        Name of the test.
    name : string
        string appended to the save file.
    results : python dictionary (optional), default = {}
        Holds the results of previous tests.
    reps : int (optional), default = 300
        How replications you want to do

    Returns
    -------
    results : python dictionary
        Dictionary that contains all the results. 
    """

    # Set-up
    # This suppresses linear svm convergence warnings.
    simplefilter(action='ignore', category=ConvergenceWarning)
    
    clf_ls = []
    clf_name = []
    
    # Random Forest
    clf_ls.append(RandomForestClassifier())
    clf_name.append("Random Forest")
    
    # Extra Trees
    clf_ls.append(ExtraTreesClassifier())
    clf_name.append("Extra Trees")
    
    # SVM - Linear
    clf = SVC(kernel="linear", max_iter=5000)
    scaler = StandardScaler()
    pipeline = Pipeline([('transformer', scaler), ('estimator', clf)])
    
    clf_ls.append(pipeline)
    clf_name.append("SVM - Linear")
    
    # SVM - RBF
    clf = SVC(kernel="rbf", max_iter=5000)
    scaler = StandardScaler()
    pipeline = Pipeline([('transformer', scaler), ('estimator', clf)])
    
    clf_ls.append(pipeline)
    clf_name.append("SVM - RBF")
    
    # Neural Network
    num_features = data.shape[1]
    num_classes = np.max(target)+1
    nn = utils.Net(structure=[num_features,100,1000,5000], num_classes=num_classes, dropout_rate=0.4,epochs=100,learning_rate=1,cuda=True)
    scaler = StandardScaler()
    pipeline = Pipeline([('transformer', scaler), ('estimator', nn)])
    
    clf_ls.append(pipeline)
    clf_name.append("Neural Network")
    
    # LDA
    clf_ls.append(LinearDiscriminantAnalysis())
    clf_name.append("Linear Discriminant Analysis")
    
    # GNB
    clf_ls.append(GaussianNB())
    clf_name.append("Gaussian Naive Bayes")
    
    # KNN
    clf = KNeighborsClassifier(n_neighbors=25) # Seemed like a good number at the time
    scalar = StandardScaler()
    pipeline = Pipeline([('transformer', scalar), ('estimator', clf)])
    
    clf_ls.append(pipeline)
    clf_name.append("K-Nearest Neighbor")
    
    r = {"Test": dataset_level+desc, "dataset": dataset_level, "reps": reps, "Specimens" : np.unique(specimens), "Results": {}}
    
    frag_ls = []
    target_ls = []
    frag_dict = {}
    component_dict = {}
    for i, frag in enumerate(specimens):
        frag_dict[frag] = {"Truth" : target[i], "Appearences" : 0}
        component_dict[frag] = {"Truth" : target[i], "Appearences" : 0}
    
    for i, clf in enumerate(clf_ls):
        name = clf_name[i]
        for frag in frag_dict.keys():
            frag_dict[frag][name] = 0
            component_dict[frag][name] = 0
        r["Results"][name] = {"iter pred": [], "iter vote": []}
    
    iter_description = "Test: " + r["Test"] + ". Reps"
    for i in trange(reps, desc=iter_description):
        data_train, target_train, data_test, target_test, frag_test = utils.train_test_split(data, target, specimens)
        for ii, clf in enumerate(clf_ls):
            name = clf_name[ii]
            clf.fit(data_train, target_train)
            pred = clf.predict(data_test)
            vote = utils.specimen_voting(pred, target_test, frag_test)
            r["Results"][name]["iter pred"].extend(pred)
            r["Results"][name]["iter vote"].append(vote)
            
        frag_ls.extend(frag_test)
        target_ls.extend(target_test)
        
        unique_frags = np.unique(frag_test)
        for frag in unique_frags:
            frag_dict[frag]["Appearences"] += 1
        
        for frag in frag_test:
            component_dict[frag]["Appearences"] += 1
        
    r["Truth Frags"] = frag_ls
    r["Truth Targets"] = target_ls
    r["Count Frags"] = frag_dict
    r["Components"] = component_dict        
    
    results[r["Test"]] = r
    
    return results

def compile_results(results):
    """Compile Results
    ========

    Compiles the results from the different tests into the forms that we can easily save. 

    Parameters
    ----------
    results : python dictionary (optional), default = {}
        Holds the results of previous tests.

    Returns
    -------
    results : python dictionary
        Dictionary that now contains everything, plus outputable lists.
    """
    # Summary and iter acc
    for test in results.keys():
        test_name = results[test]["Test"]
        reps = results[test]["reps"]
        
        # Component Confusion matrices & f1 scores
        target_truth = results[test]["Truth Targets"]
        for algo in results[test]["Results"].keys():
            component_cm = confusion_matrix(target_truth, results[test]["Results"][algo]["iter pred"])
            results[test]["Results"][algo]["Component CM"] = component_cm
            
            component_f1 = f1_score(target_truth,results[test]["Results"][algo]["iter pred"])
            results[test]["Results"][algo]["Component F1"] = component_f1
        
        # Appearances
        for algo in results[test]["Results"].keys():
            for mesh in results[test]["Truth Frags"]:
                results[test]["Components"][mesh][algo] = 0
                results[test]["Count Frags"][mesh][algo] = 0
            
            # Components
            for i, pred in enumerate(results[test]["Results"][algo]["iter pred"]):
                mesh = results[test]["Truth Frags"][i]
                truth = results[test]["Truth Targets"][i]
                results[test]["Components"][mesh][algo] += 1 * (pred == truth)
            
            # Voting
            vote_pred = []
            vote_truth = []
            for vote in results[test]["Results"][algo]["iter vote"]:
                for mesh in vote["frags"].keys():
                    guess = vote["frags"][mesh]["Guess"]
                    truth = vote["frags"][mesh]["Truth"]
                    results[test]["Count Frags"][mesh][algo] += 1 * (guess == truth)
                    vote_pred.append(guess)
                    vote_truth.append(truth)
            
            # Voting CM
            vote_cm = confusion_matrix(vote_truth, vote_pred)
            results[test]["Results"][algo]["Vote CM"] = vote_cm
            
            # Voting F1
            vote_f1 = f1_score(vote_truth, vote_pred)
            results[test]["Results"][algo]["Vote F1"] = vote_f1
            
        # Confusion Matrices Again
            
        saved_component_cm = [["Test:", test_name, "Component CM", "Reps:", reps]]
        saved_component_cm.append(["Algorithm", "Confusion Matrix"])
        
        saved_vote_cm = [["Test:", test_name, "Vote CM", "Reps:", reps]]
        saved_vote_cm.append(["Algorithm", "Confusion Matrix"])
        
        for algo in results[test]["Results"].keys():
            name = algo
            
            cm = results[test]["Results"][algo]["Component CM"]
            saved_component_cm.append([name])
            saved_component_cm.extend(cm)
            saved_component_cm.append([]) # Spacer
            
            cm = results[test]["Results"][algo]["Vote CM"]
            saved_vote_cm.append([name])
            saved_vote_cm.extend(cm)
            saved_vote_cm.append([]) # Spacer
            
        results[test]["Saved Component CM"] = saved_component_cm
        results[test]["Saved Vote CM"] = saved_vote_cm
        
        # Appearances Again
        component_apperences = [["Test:", test_name, "Component Appearances", "Reps:", reps]]
        vote_apperences = [["Test:", test_name, "Vote Appearances", "Reps:", reps]]
        algos = [*results[test]["Results"].keys()]
        header = algos.copy()
        header.insert(0, "Appearences")
        header.insert(0, "Specimen")
        component_apperences.append(header)
        vote_apperences.append(header)
        algos.insert(0, "Appearences")
        
        for mesh in results[test]["Specimens"]:
            component_inner_ls = [mesh]
            vote_inner_ls = [mesh]
            for algo in algos:
                component_inner_ls.append(results[test]["Components"][mesh][algo])
                vote_inner_ls.append(results[test]["Count Frags"][mesh][algo])
            component_apperences.append(component_inner_ls)
            vote_apperences.append(vote_inner_ls)
        
        results[test]["Saved Components"] = component_apperences
        results[test]["Saved Votes"] = vote_apperences
    
        save_sum_ls = [["Test:", test_name, "Summary Stats", "Reps:", reps]]
        save_sum_ls.append(["Algorithm", "Mean Accuracy", "Standard Deviation", "F1 Score", "Component F1 Score"])
        for algo in results[test]["Results"].keys():
            inner_ls = []
            for vote in results[test]["Results"][algo]["iter vote"]:
                inner_ls.append(vote["Mean Accuracy"])
            results[test]["Results"][algo]["saved iter acc"] = inner_ls
            save_sum_ls.append([algo, 100 * np.mean(inner_ls), 100 * np.std(inner_ls), results[test]["Results"][algo]["Vote F1"], results[test]["Results"][algo]["Component F1"]])
        results[test]["Summary Acc"] = save_sum_ls
        
        # Iter Acc
        save_iter_ls = [["Test:", test_name, "Iter Accuracy", "Reps:", reps]]
        save_iter_ls.append([*results[test]["Results"].keys()])
        names_inner_ls = []
        iter_inner_ls = []
        for algo in results[test]["Results"].keys():
            names_inner_ls.append(algo)
            iter_inner_ls.append(results[test]["Results"][algo]["saved iter acc"])
        
        save_iter_ls.append(names_inner_ls)
        
        # There are better ways to reshape these lists
        for i in range(reps):
            inner_ls = []
            for ii in range(len(iter_inner_ls)):
                inner_ls.append(iter_inner_ls[ii][i])
            save_iter_ls.append(inner_ls)
        
        results[test]["Iter Algo Acc"] = save_iter_ls
    
    return results

def save_results(results):
    """Save Results
    ========

    Saves the results from the tests in multiple csv files. 

    Parameters
    ----------
    results : python dictionary
        Dictionary that contains all the results.
    """
    
    dt = datetime.now() # So you don't accidentally lose data if you happen to have a previous test thing open.
    
    # Datetime doesn't store things as 2 digit strings, but I want them in that format. 
    if dt.month < 10:
        month = f"0{dt.month}"
    else:
        month = dt.month
    
    if dt.day < 10:
        day =  f"0{dt.day}"
    else:
        day = dt.day
      
    if dt.hour < 10:
        hour = f"0{dt.hour}"
    else:
        hour = dt.hour    
      
    if dt.minute < 10:
        minute = f"0{dt.minute}"
    else:
        minute = dt.minute
    
    header = f"./results/AMMAZE Tests {dt.year}{month}{day}{hour}{minute}/"
    os.mkdir(header)
    
    fname = "Summary_"
    
    footer = ".csv"
    
    for test in results.keys():
        test_name = results[test]["Test"]
        with open(header+fname+test_name+footer, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(results[test_name]["Summary Acc"])
    
    fname = "Iter_Acc_"
    
    for test in results.keys():
        test_name = results[test]["Test"]
        with open(header+fname+test_name+footer, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(results[test_name]["Iter Algo Acc"])       
            
    fname = "Confusion_Matrices_"
    
    for test in results.keys():
        test_name = results[test]["Test"]
        with open(header+fname+test_name+footer, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if test_name == "frags":
                # Components will equal votes in this case, so we don't save them
                writer.writerows(results[test_name]["Saved Vote CM"])
            else:
                writer.writerows(results[test_name]["Saved Vote CM"])
                writer.writerows(results[test_name]["Saved Component CM"])
      
    fname = "Frag_Predictions_"
    
    for test in results.keys():
        dataset_level = results[test]["dataset"]
        test_name = results[test]["Test"]
        if dataset_level == "frags":
            # Components will equal votes in this case, so we don't save them
            with open(header+fname+test_name+footer, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(results[test_name]["Saved Votes"])
        else:
            with open(header+fname+test_name+footer, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(results[test_name]["Saved Votes"])
                
            # Components are different, and should be saved in a seperate file
            with open(header+fname+test_name+"_Components"+footer, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(results[test_name]["Saved Components"])
                
def main():
    # Main Break Test
    break_test = {"dataset": "breaks", "target_field": "Effector", "desc": "","numerical_fields" : None, "categorical_fields" : None, "sum_stats_fields" : None, "sum_stats" : None, "count_fields" : None}
    
    # Main Frag Test
    frag_test = {"dataset": "frags", "target_field": "Effector", "desc": "",
                 "numerical_fields" : None, "categorical_fields" : None, "sum_stats_fields" : None, "sum_stats" : None, "count_fields" : None}
    
    # To load the data with only fragment level features:
    # frag_test_frag_only = {"dataset": "frags", "target_field": "Effector", "desc": "_frag_level_vars_only",
                           # "numerical_fields" : None, "categorical_fields" : None, "sum_stats_fields" : [], "sum_stats" : [], "count_fields" : []}    
    
    # To get the data with only break level features:
    # frag_test_break_only = {"dataset": "frags", "target_field": "Effector", "desc": "_break_level_vars_only", "numerical_fields" : [], "categorical_fields" : [] , "sum_stats_fields" : None, "sum_stats" : None, "count_fields" : None}    
    
    # Further Tests...
    # Note that if your test desc are the same across datasets, things will overwrite because the test name will be the same, and you probably don't want that. 
    
    # optional_test = {"dataset" : "breaks", "target_field" : "Effector", "desc": "AMAAAZING",
    #                  "numerical_fields" : [], "categorical_fields" : [], "sum_stats_fields" : [], "sum_stats" : [], "count_fields" : []}
    
    # Test compilation
    tests = [break_test]
    
    # Iterating over the tests
    results = {}
    for test in tests:
        data, target, specimens = create_dataset(test)
        results = run_test(data, target, specimens, test["dataset"], test["desc"])
        
    results = compile_results(results)
    
    save_results(results)


if __name__ == '__main__':
    main()
