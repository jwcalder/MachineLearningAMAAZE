import utils
import csv
import numpy as np
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from tqdm import trange


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


def run_test(data, target, specimens, test_name, results={}, reps=300, rf_nb = False):
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
    rf_nb : boolean (optional), default = False
        If you want to run an additional bootstrapless random forest model

    Returns
    -------
    results : python dictionary
        Dictionary that contains all the results. 
    """

    # Yes, we vote on fragments even if fragments are being classified on (thus, there's only one vote) to make the code cleaner. It will not change the results.

    # Set-up
    # This suppresses linear svm convergence warnings.
    simplefilter(action='ignore', category=ConvergenceWarning)

    r = {"Test": test_name, "reps": reps, "Results": {}}

    # Not the most elegant solution, but these lists will hold the accuracy and predictions for each rep
    rf_acc = np.empty(reps)
    et_acc = np.empty(reps)
    svmL_acc = np.empty(reps)
    svmRBF_acc = np.empty(reps)
    nn_acc = np.empty(reps)
    lda_acc = np.empty(reps)
    gnb_acc = np.empty(reps)
    knn_acc = np.empty(reps)
    
    rf_target_pred = []
    et_target_pred = []
    svmL_target_pred = []
    svmRBF_target_pred = []
    nn_target_pred = []
    lda_target_pred = []
    gnb_target_pred = []
    knn_target_pred = []
     
    # We create these, but don't use them unless they're activated
    rf_nb_acc = np.empty(reps)
    rf_nb_target_pred = []
    breaks_rf_nb_target_pred = []
    
    breaks_rf_target_pred = []
    breaks_et_target_pred = []
    breaks_svmL_target_pred = []
    breaks_svmRBF_target_pred = []
    breaks_nn_target_pred = []
    breaks_lda_target_pred = []
    breaks_gnb_target_pred = []
    breaks_knn_target_pred = []
    
    target_test_frags = []
    target_truth = []
    
    breaks_target_test_frags = []
    breaks_target_truth = []
    
    # Global Frag to guess conversion apoclapse or something
    frag_to_guess = {}
    breaks_frag_to_guess = {}
    for i, frag in enumerate(specimens):
        frag_to_guess[frag] = {"Truth" : target[i], "Random Forest": 0, "Extra Trees" : 0, "SVM - linear" : 0, "SVM - RBF" : 0, "Neural Network" : 0, "LDA" : 0, "Naive Bayes" : 0, "KNN" : 0, "Random Forest (no bootstrap)" : 0, "Appearences" : 0}
        breaks_frag_to_guess[frag] = {"Truth" : target[i], "Random Forest": 0, "Extra Trees" : 0, "SVM - linear" : 0, "SVM - RBF" : 0, "Neural Network" : 0, "LDA" : 0, "Naive Bayes" : 0, "KNN" : 0, "Random Forest (no bootstrap)" : 0, "Appearences" : 0}
    
    iter_description = "Test: " + test_name + ". Reps"
    for i in trange(reps, desc=iter_description):
        data_train, target_train, data_test, target_test, frag_test = utils.train_test_split(
            data, target, specimens)

        # The Tests

        # Random Forest
        clfrf = RandomForestClassifier()
        clfrf.fit(data_train, target_train)
        rf_pred = clfrf.predict(data_test)
        rf_vote = utils.specimen_voting(rf_pred, target_test, frag_test)
        rf_acc[i] = 100 * rf_vote["Mean Accuracy"]
        breaks_rf_target_pred.extend(rf_pred)
        
        # Extra Trees
        clfet = ExtraTreesClassifier()
        clfet.fit(data_train, target_train)
        et_pred = clfet.predict(data_test)
        et_vote = utils.specimen_voting(et_pred, target_test, frag_test)
        et_acc[i] = 100 * et_vote["Mean Accuracy"]
        breaks_et_target_pred.extend(et_pred)
        
        # SVM - Linear
        clfsvmL = SVC(kernel="linear", max_iter=5000)
        scalar = StandardScaler()  # Some algorithms work best with standard scaled data
        # That's what this does, in a safe manner
        pipeline = Pipeline([('transformer', scalar), ('estimator', clfsvmL)])
        pipeline.fit(data_train, target_train)
        svmL_pred = pipeline.predict(data_test)
        svmL_vote = utils.specimen_voting(svmL_pred, target_test, frag_test)
        svmL_acc[i] = 100 * svmL_vote["Mean Accuracy"]
        breaks_svmL_target_pred.extend(svmL_pred)
        
        # SVM - RBF
        clfsvmRBF = SVC(kernel="rbf", max_iter=5000)
        scalar = StandardScaler()
        pipeline = Pipeline([('transformer', scalar), ('estimator', clfsvmRBF)])
        pipeline.fit(data_train, target_train)
        svmRBF_pred = pipeline.predict(data_test)
        svmRBF_vote = utils.specimen_voting(svmRBF_pred, target_test, frag_test)
        svmRBF_acc[i] = 100 * svmRBF_vote["Mean Accuracy"]
        breaks_svmRBF_target_pred.extend(svmRBF_pred)
        
        # Neural Network
        num_features = data.shape[1]
        num_classes = np.max(target)+1
        nn = utils.Net(structure=[num_features,100,1000,5000], num_classes=num_classes, dropout_rate=0.4,epochs=100,learning_rate=1,cuda=True)
        nn_scalar = StandardScaler().fit(data_train)  # Scaling data
        nn.fit(nn_scalar.transform(data_train), target_train)
        nn_pred = nn.predict(nn_scalar.transform(data_test))
        nn_vote = utils.specimen_voting(nn_pred, target_test, frag_test)
        nn_acc[i] = 100 * nn_vote["Mean Accuracy"]
        breaks_nn_target_pred.extend(nn_pred)
        
        # LDA
        clfLDA = LinearDiscriminantAnalysis()
        clfLDA.fit(data_train, target_train)
        lda_pred = clfLDA.predict(data_test)
        lda_vote = utils.specimen_voting(lda_pred, target_test, frag_test)
        lda_acc[i] = 100 * lda_vote["Mean Accuracy"]
        breaks_lda_target_pred.extend(lda_pred)
        
        # Naive Bayes
        clfgnb = GaussianNB()
        clfgnb.fit(data_train, target_train)
        gnb_pred = clfgnb.predict(data_test)
        gnb_vote = utils.specimen_voting(gnb_pred, target_test, frag_test)
        gnb_acc[i] = 100 * gnb_vote["Mean Accuracy"]
        breaks_gnb_target_pred.extend(gnb_pred)
        
        # KNN
        clfknn = KNeighborsClassifier(n_neighbors=25) # Seemed like a good number at the time
        scalar = StandardScaler()
        pipeline = Pipeline([('transformer', scalar), ('estimator', clfknn)])
        pipeline.fit(data_train, target_train)
        knn_pred = pipeline.predict(data_test)
        knn_vote = utils.specimen_voting(knn_pred, target_test, frag_test)
        knn_acc[i] = 100 * knn_vote["Mean Accuracy"]
        breaks_knn_target_pred.extend(knn_pred)

        if(rf_nb):
            # Random Forest no bootstrap
            clfrf_nb = RandomForestClassifier(bootstrap = False)
            clfrf_nb.fit(data_train, target_train)
            rf_nb_pred = clfrf_nb.predict(data_test)
            rf_nb_vote = utils.specimen_voting(rf_nb_pred, target_test, frag_test)
            rf_nb_acc[i] = 100 * rf_nb_vote["Mean Accuracy"]
            breaks_rf_nb_target_pred.extend(rf_nb_pred)
        
        # Cleanup
        
        breaks_target_truth.extend(target_test)
        
        # Predictions per algorithm
        unique_frags = list(set(frag_test))
        for frag in unique_frags:
            frag_to_guess[frag]["Appearences"] += 1
            frag_to_guess[frag]["Random Forest"] += 1 * (rf_vote["frags"][frag]["Guess"] == frag_to_guess[frag]["Truth"])
            frag_to_guess[frag]["Extra Trees"] += 1 * (et_vote["frags"][frag]["Guess"] == frag_to_guess[frag]["Truth"])
            frag_to_guess[frag]["SVM - linear"] += 1 * (svmL_vote["frags"][frag]["Guess"] == frag_to_guess[frag]["Truth"])
            frag_to_guess[frag]["SVM - RBF"] += 1 * (svmRBF_vote["frags"][frag]["Guess"] == frag_to_guess[frag]["Truth"])
            frag_to_guess[frag]["Neural Network"] += 1 * (nn_vote["frags"][frag]["Guess"] == frag_to_guess[frag]["Truth"])
            frag_to_guess[frag]["Naive Bayes"] += 1 * (gnb_vote["frags"][frag]["Guess"] == frag_to_guess[frag]["Truth"])
            frag_to_guess[frag]["LDA"] += 1 * (lda_vote["frags"][frag]["Guess"] == frag_to_guess[frag]["Truth"])
            frag_to_guess[frag]["KNN"] += 1 * (knn_vote["frags"][frag]["Guess"] == frag_to_guess[frag]["Truth"])
            
            # Confusion matrix stuff
            target_truth.append(frag_to_guess[frag]["Truth"])
            rf_target_pred.append(rf_vote["frags"][frag]["Guess"])
            et_target_pred.append(et_vote["frags"][frag]["Guess"])
            svmL_target_pred.append(svmL_vote["frags"][frag]["Guess"])
            svmRBF_target_pred.append(svmRBF_vote["frags"][frag]["Guess"])
            nn_target_pred.append(nn_vote["frags"][frag]["Guess"])
            lda_target_pred.append(lda_vote["frags"][frag]["Guess"])
            gnb_target_pred.append(gnb_vote["frags"][frag]["Guess"])
            knn_target_pred.append(knn_vote["frags"][frag]["Guess"])
           
            if(rf_nb):
                frag_to_guess[frag]["Random Forest (no bootstrap)"] += 1 * (rf_nb_vote["frags"][frag]["Guess"] == frag_to_guess[frag]["Truth"])
                rf_nb_target_pred.append(rf_nb_vote["frags"][frag]["Guess"])                
        
        # This will give us a picture of the underlying predictions
        for i, frag in enumerate(frag_test):
            breaks_frag_to_guess[frag]["Appearences"] += 1
            breaks_frag_to_guess[frag]["Random Forest"] += 1 * (rf_pred[i] == target_test[i])
            breaks_frag_to_guess[frag]["Extra Trees"] += 1 * (et_pred[i] == target_test[i])
            breaks_frag_to_guess[frag]["SVM - linear"] += 1 * (svmL_pred[i] == target_test[i])
            breaks_frag_to_guess[frag]["SVM - RBF"] += 1 * (svmRBF_pred[i] == target_test[i])
            breaks_frag_to_guess[frag]["Neural Network"] += 1 * (nn_pred[i] == target_test[i])
            breaks_frag_to_guess[frag]["Naive Bayes" ] += 1 * (gnb_pred[i] == target_test[i])
            breaks_frag_to_guess[frag]["LDA"] += 1 * (lda_pred[i] == target_test[i])
            breaks_frag_to_guess[frag]["KNN"] += 1 * (knn_pred[i] == target_test[i])
           
            if(rf_nb):
                breaks_frag_to_guess[frag]["Random Forest (no bootstrap)"] += 1 * (rf_nb_pred[i] == target_test[i])
            
    
    # Confusion matrix creation
    rf_cm = confusion_matrix(target_truth, rf_target_pred)
    et_cm = confusion_matrix(target_truth, et_target_pred)
    svmL_cm = confusion_matrix(target_truth, svmL_target_pred)
    svmRBF_cm = confusion_matrix(target_truth, svmRBF_target_pred)
    nn_cm = confusion_matrix(target_truth, nn_target_pred)
    lda_cm = confusion_matrix(target_truth, lda_target_pred)
    gnb_cm = confusion_matrix(target_truth, gnb_target_pred)
    knn_cm = confusion_matrix(target_truth, knn_target_pred)
    
    breaks_rf_cm = confusion_matrix(breaks_target_truth, breaks_rf_target_pred)
    breaks_et_cm = confusion_matrix(breaks_target_truth, breaks_et_target_pred)
    breaks_svmL_cm = confusion_matrix(breaks_target_truth, breaks_svmL_target_pred)
    breaks_svmRBF_cm = confusion_matrix(breaks_target_truth, breaks_svmRBF_target_pred)
    breaks_nn_cm = confusion_matrix(breaks_target_truth, breaks_nn_target_pred)
    breaks_lda_cm = confusion_matrix(breaks_target_truth, breaks_lda_target_pred)
    breaks_gnb_cm = confusion_matrix(breaks_target_truth, breaks_gnb_target_pred)
    breaks_knn_cm = confusion_matrix(breaks_target_truth, breaks_knn_target_pred)
    
    if(rf_nb):
        rf_nb_cm = confusion_matrix(target_truth, rf_nb_target_pred)
        breaks_rf_nb_cm = confusion_matrix(breaks_target_truth, breaks_rf_nb_target_pred)
        r["Results"]["Random Forest (no bootstrap)"] = {"accuracy": np.mean(rf_nb_acc), "std": np.std(rf_nb_acc), "iter accuracy": rf_nb_acc, "vote pred": rf_nb_target_pred, "component vote pred": breaks_rf_nb_target_pred, "confusion matrix": rf_nb_cm, "component confusion matrix": breaks_rf_nb_cm}
    
    r["Results"]["Random Forest"] = {"accuracy": np.mean(rf_acc), "std": np.std(rf_acc), "iter accuracy": rf_acc, "vote pred":  rf_target_pred, "component vote pred": breaks_rf_target_pred, "confusion matrix": rf_cm, "component confusion matrix": breaks_rf_cm}
    r["Results"]["Extra Trees"] = {"accuracy": np.mean(et_acc), "std": np.std(et_acc), "iter accuracy": et_acc, "vote pred": et_target_pred, "component vote pred": breaks_et_target_pred,"confusion matrix": et_cm, "component confusion matrix": breaks_et_cm}
    r["Results"]["SVM - linear"] = {"accuracy": np.mean(svmL_acc), "std": np.std(svmL_acc), "iter accuracy": svmL_acc, "vote pred": svmL_target_pred, "component vote pred": breaks_svmL_target_pred,"confusion matrix": svmL_cm, "component confusion matrix": breaks_svmL_cm}
    r["Results"]["SVM - RBF"] = {"accuracy": np.mean(svmRBF_acc), "std": np.std(svmRBF_acc), "iter accuracy": svmRBF_acc, "vote pred": svmRBF_target_pred, "component vote pred": breaks_svmRBF_target_pred, "confusion matrix": svmRBF_cm, "component confusion matrix": breaks_svmRBF_cm}
    r["Results"]["Neural Network"] = {"accuracy": np.mean(nn_acc), "std": np.std(nn_acc), "iter accuracy": nn_acc, "vote pred": nn_target_pred, "component vote pred": breaks_nn_target_pred, "confusion matrix": nn_cm, "component confusion matrix": breaks_nn_cm}
    r["Results"]["LDA"] = {"accuracy": np.mean(lda_acc), "std": np.std(lda_acc), "iter accuracy": lda_acc, "vote pred": lda_target_pred, "component vote pred": breaks_lda_target_pred,"confusion matrix": lda_cm, "component confusion matrix": breaks_lda_cm}
    r["Results"]["Naive Bayes"] = {"accuracy": np.mean(gnb_acc), "std": np.std(gnb_acc), "iter accuracy": gnb_acc, "vote pred": gnb_target_pred, "component vote pred": breaks_gnb_target_pred, "confusion matrix": gnb_cm, "component confusion matrix": breaks_gnb_cm}
    r["Results"]["KNN"] = {"accuracy": np.mean(knn_acc), "std": np.std(knn_acc), "iter accuracy": knn_acc, "vote pred": knn_target_pred, "component vote pred": breaks_knn_target_pred, "confusion matrix": knn_cm, "component confusion matrix": breaks_knn_cm}
    r["Test Frags"] = target_test_frags
    r["Guess by Frag"] = frag_to_guess
    r["Component Test Frags"] = breaks_target_test_frags
    r["Component Guess by Frag"] = breaks_frag_to_guess
    
    results[test_name] = r
    
    return results


def save_results(results, fname="AMAAZE_test_results.csv", full=False, full_header="AMAAZE_full_results_"):
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

    fname = "./results/"+fname

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
        # Iter accuracy
        for test in results.keys():
            test_name = results[test]["Test"]
            reps = results[test]["reps"]
            fname = "./results/"+full_header + test_name + "_iter_acc" + ".csv"
            with open(fname, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Test: " + test_name, "Reps:", reps])
                # Column for each algorithm
                writer.writerow(results[test]["Results"].keys())
                data = []
                for algo in results[test]["Results"].keys():
                    data.append(results[test]["Results"][algo]["iter accuracy"])
                rows = zip(*data)
                writer.writerows(rows)
            
        # Guesses across fragments per algorithm
        for test in results.keys():
            test_name = results[test]["Test"]
            reps = results[test]["reps"]
            fname = "./results/"+full_header + test_name + "_frag_guesses" + ".csv"
            with open(fname, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Test: " + test_name, "Reps:", reps, "Numbers correspond to the number of times that algorithm was able to correctly classify that specimen"])
                algos = [*results[test]["Results"].keys()] # Dictionary keys are a black hole of type errors. Iterable errors. What are they? Only the interpreter knows.
                header = algos.copy()
                header.insert(0, "Appearences")
                header.insert(0, "Specimen")
                algos.insert(0, "Appearences")
                writer.writerow(header)
                for frag in results[test]["Guess by Frag"].keys():
                    collect = [str(frag)]
                    for algo in algos:
                        collect.append(results[test]["Guess by Frag"][frag][algo])
                    writer.writerow(collect)
            
            if test_name == "breaks": # Components and votes will match in the fragment level case.
                fname = "./results/"+ full_header + test_name + "_component_frag_guesses" + ".csv"
                with open(fname, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Test: " + test_name, "Reps:", reps, "Numbers correspond to the number of times that algorithm was able to correctly classify that specimen before voting"])
                    algos = [*results[test]["Results"].keys()] # Dictionary keys are a black hole of type errors. Iterable errors. What are they? Only the interpreter knows.
                    header = algos.copy()
                    header.insert(0, "Appearences")
                    header.insert(0, "Specimen")
                    algos.insert(0, "Appearences")
                    writer.writerow(header)
                    for frag in results[test]["Component Guess by Frag"].keys():
                        collect = [str(frag)]
                        for algo in algos:
                            collect.append(results[test]["Component Guess by Frag"][frag][algo])
                        writer.writerow(collect)
            
        # Confusion Matrices
        for test in results.keys():
            test_name = results[test]["Test"]
            reps = results[test]["reps"]
            fname = "./results/"+full_header + test_name + "_confusion_matrix" + ".csv"
            with open(fname, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Test: " + test_name, "Reps:", reps])
                writer.writerow(["Algorithm", "Confusion Matrix"])
                for algo in results[test]["Results"].keys():
                    name = algo
                    cm = results[test]["Results"][algo]["confusion matrix"]
                    writer.writerow([name])
                    writer.writerows(cm)
                writer.writerow([])
                
            if test_name == "breaks":
                fname = "./results/" + full_header + test_name + "_component_confusion_matrix" + ".csv"
                with open(fname, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Test: " + test_name, "Reps:", reps])
                    writer.writerow(["Algorithm", "Component Confusion Matrix"])
                    for algo in results[test]["Results"].keys():
                        name = algo
                        cm = results[test]["Results"][algo]["component confusion matrix"]
                        writer.writerow([name])
                        writer.writerows(cm)
                    writer.writerow([])


def main():
    tests = ["breaks", "frags"] # "breaks", "frags"
    results = {}
    for test in tests:
        data, target, specimens = create_dataset(test)
        results = run_test(data, target, specimens, test, rf_nb = True)

    save_results(results, full=True)


if __name__ == '__main__':
    main()
