from sklearn import svm
import csv
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from tqdm import trange
import scipy.stats as sci_ST
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, RepeatedKFold
import numpy as np
from sklearn import preprocessing, linear_model
import statistics as ST
import pandas as pd
import ssl
#import nn3
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score
#from sklearn.pipeline import make_pipeline
#from sklearn import metrics


ssl._create_default_https_context = ssl._create_unverified_context


df = pd.read_csv('GeneralStats.csv') # or whatever your dataset is called
df

"""Preprocess data and labels to numerical values.

"""

# Labels
le = preprocessing.LabelEncoder()
labels = le.fit_transform(df['effector'])

# Categorical Data
cat_data = df[[]]
le = preprocessing.OneHotEncoder(sparse=False)
cat_data = le.fit_transform(cat_data)

# This is the header of our csv_file (except effector, which we're classfiying on)
header_list = [
    "freq_obtuse",
    "freq_right",
    "freq_acute",
    "b1",
    "b2",
    "b3",
    "SA",
    "Vol",
    "a_min", "a_max", "a_mean", "a_median", "a_stdv",
    "bkm_min", "bkm_max", "bkm_mean", "bkm_median", "bkm_stdv",
    "bkp_min", "bkp_max", "bkp_mean", "bkp_median", "bkp_stdv"
]

# Numerical data
num_data = df[header_list].values  # To use only break level data

# Combine Categorical and numerical data
data = np.hstack((cat_data, num_data))
print(data.shape)

frag_name = df['frag_name']

"""Now we do a train test split and apply some machine learning."""


# Folds for K-Fold cross validations
reps = 3
folds = 10

scalar = StandardScaler()

### This is our cross validation index thing
cv = RepeatedKFold(n_splits = folds, n_repeats = reps)

### Random Forests
rf = RandomForestClassifier()
rf_acc = 100*cross_val_score(rf, data, labels, cv = cv)
print("RF: %.2f%%"%ST.mean(rf_acc))

### Support Vector Machine
clfsvm = SVC(kernel ="linear", max_iter = 2000) # "linear"
pipeline = Pipeline([('transformer', scalar), ('estimator', clfsvm)])
svm_acc = 100*cross_val_score(pipeline, data, labels, cv = cv, n_jobs = -1) # The n_jobs -1 means it will max out your computer's processor to do this as fast as possible.
print("SVM: %.2f%%"%ST.mean(svm_acc))

### K-Nearest Neighbors
k = int(50) # 50 seems to work with our current dataset size and the standard scaler.
clfNN = KNeighborsClassifier(n_neighbors=k)
pipeline = Pipeline([('transformer', scalar), ('estimator', clfNN)])
KNN_acc = 100*cross_val_score(pipeline, data, labels, cv = cv)
print("KNN: %.2f%%"%ST.mean(KNN_acc))

### Linear Discriminant Analysis
clfLDA = LinearDiscriminantAnalysis()
LDA_acc = 100*cross_val_score(clfLDA, data, labels, cv = cv)
print("LDA: %.2f%%"%ST.mean(LDA_acc))

# ### Graph based learning etc.
# Copy the structure of the above algorithms, and it should *just work*
X = df.iloc[:,2:]
X = np.array(X)#Load data
k = 20 #number of neighbors
W = GraphLearning.getMatrix(X, k)#Scale Data and get matrix.
GL_classifier = GraphLearning(W, k, 'laplace')
cv = RepeatedKFold(n_splits = 4, n_repeats = reps)
data = np.arange(X.shape[0])
gl_acc = 100*cross_val_score(GL_classifier, data, labels, cv = cv, scoring='accuracy')
print(gl_acc)
print(np.mean(gl_acc))


# AHHHHHHHHHHH
# Okay, uncommenting everything below here will create a 
# spreadsheet of the accuracy and confidence across all fragments

# tmp_list = []

# for index in range(len(labels)):
#     tmp_list.append([frag_name[index], labels[index], [], [], 0, 0])
#     # Structure is:
#     # [Frag name, true label, confidence when correct, confidence when incorrect, times correct, times incorrect]

# correct_probs = []
# incorrect_probs = []

# with trange(reps, desc="Reps") as t:
#     for i in t:
#         cv = KFold(n_splits=folds, random_state=None, shuffle=True)  # 10

#         # Random Forests
#         rf = RandomForestClassifier()
#         rf_predict = cross_val_predict(
#             rf, data, labels, cv=cv, method='predict_proba', n_jobs=-1)

#         # Linear Discriminant Analysis
#         clfLDA = LinearDiscriminantAnalysis()
#         LDA_predict = cross_val_predict(
#             clfLDA, data, labels, cv=cv, method='predict_proba', n_jobs=-1)

#         # K-Nearest Neighbors
#         k = int(50)  # Our total fragment number after rebalancing. Was 50
#         clfNN = KNeighborsClassifier(n_neighbors=k)
#         pipeline = Pipeline([('transformer', scalar), ('estimator', clfNN)])
#         KNN_predict = cross_val_predict(
#             pipeline, data, labels, cv=cv, method='predict_proba', n_jobs=-1)

#         # Support Vector Machine
#         clfsvm = SVC(kernel="linear", max_iter=5000,
#                      probability=True)  # "linear"
#         pipeline = Pipeline([('transformer', scalar), ('estimator', clfsvm)])
#         svm_predict = cross_val_predict(
#             pipeline, data, labels, cv=cv, method='predict_proba', n_jobs=-1)

#         for i in range(len(labels)):
#             true_label = labels[i]

#             # All of our classifer's confidence scores in their predictions
#             rf_prob0, rf_prob1 = rf_predict[i]
#             LDA_prob0, LDA_prob1 = LDA_predict[i]
#             KNN_prob0, KNN_prob1 = KNN_predict[i]
#             svm_prob0, svm_prob1 = svm_predict[i]

#             # these are for if we want to add more classifiers
#             prob_0_list = [rf_prob0, LDA_prob0, KNN_prob0, svm_prob0]
#             prob_1_list = [rf_prob1, LDA_prob1, KNN_prob1, svm_prob1]

#             if true_label == 0:
#                 for ii in range(len(prob_0_list)):
#                     prob0 = prob_0_list[ii]
#                     prob1 = prob_1_list[ii]
#                     if prob0 > 0.5:
#                         correct_probs.append(prob0)
#                         tmp_list[i][2].append(prob0)
#                         tmp_list[i][4] += 1
#                     else:
#                         incorrect_probs.append(prob1)
#                         tmp_list[i][3].append(prob1)
#                         tmp_list[i][5] += 1
#             else:
#                 for ii in range(len(prob_0_list)):
#                     prob0 = prob_0_list[ii]
#                     prob1 = prob_1_list[ii]
#                     if prob1 > 0.5:
#                         correct_probs.append(prob1)
#                         tmp_list[i][2].append(prob1)
#                         tmp_list[i][4] += 1
#                     else:
#                         incorrect_probs.append(prob0)
#                         tmp_list[i][3].append(prob0)
#                         tmp_list[i][5] += 1


# # Confidence intervals
# correct_probs = sci_ST.t.interval(0.95, len(
#     correct_probs)-1, loc=np.mean(correct_probs), scale=sci_ST.sem(correct_probs))
# incorrect_probs = sci_ST.t.interval(0.95, len(
#     incorrect_probs)-1, loc=np.mean(incorrect_probs), scale=sci_ST.sem(incorrect_probs))

# CI_C_L, CI_C_U = correct_probs
# CI_IC_L, CI_IC_U = incorrect_probs

# CI_C_L = (str(CI_C_L))[:5]
# CI_C_U = (str(CI_C_U))[:5]
# CI_IC_L = (str(CI_IC_L))[:5]
# CI_IC_U = (str(CI_IC_U))[:5]

# CI_C = "[" + CI_C_L + ", " + CI_C_U + "]"
# CI_IC = "[" + CI_IC_L + ", " + CI_IC_U + "]"
# ####

# output_list = tmp_list

# for index in range(len(tmp_list)):
#     if len(tmp_list[index][2]) == 0:
#         output_list[index][2] = 0
#     else:
#         output_list[index][2] = ST.mean(tmp_list[index][2])
#     if len(tmp_list[index][3]) == 0:
#         output_list[index][3] = 0
#     else:
#         output_list[index][3] = ST.mean(tmp_list[index][3])


# # output_list = []
# # for index in range(len(labels)):
# #     if num_list[index] > 0:
# #         output_list.append([frag_name_ordered[index],num_list[index]])

# header_list = ["Frag name", "True label",
#                "AVG Prob when correct", "AVG Prob when incorrect",
#                "Times correct", "Times incorrect",
#                "95% CI for correct", CI_C, "95% CI for incorrect", CI_IC]
# output_file = "probs.csv"
# with open(output_file, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(header_list)
#     writer.writerows(output_list)
