from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, RepeatedKFold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ssl
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

ssl._create_default_https_context = ssl._create_unverified_context

df = pd.read_csv('moclan.csv')
# df = pd.read_csv('moclan_carngrouped.csv')

### Bootstrap
# boot_reps = 1000 # How many copies we want
# boot_strap = df.sample(boot_reps, replace=True)
# df = pd.concat([df, boot_strap])

### Labels
le = preprocessing.LabelEncoder()
labels = le.fit_transform(df['Agent'])

### Categorical Data
# Note that Epiphysis, Notch_a, and Notch_c all contain levels beyond "Absent" and "Present"
# This is unexpected for a boolean variable. Those variables are ommitted in the "clean" vector. 
# Number of planes can be treated as a catagorical variable or a numerical variable, because it has so few levels. 
# The purpose and level of the ">4cm" variable is unclear, so we've ommited it from both the fragment and break level tests.
# Type of angle is redundent with angle 

# cat_data = df[['Epiphysis', 'Number_of_planes', 'Type_of_angle', 'Notch', 'Notch_a', 'Notch_c', 'Notch_d', 'Fracture_plane', '>4cm', 'Interval(length)']] # Full data
# cat_data = df[['Number_of_planes', 'Type_of_angle', 'Notch', 'Notch_d', 'Fracture_plane', '>4cm', 'Interval(length)']] # Full data without unclean data
# cat_data = df[['Number_of_planes','Notch','Notch_d','Interval(length)']] # Clean Frag only
cat_data = df[['Fracture_plane']] # Clean Break level only
# cat_data = df[[]]  # No cat vars

le = preprocessing.OneHotEncoder(sparse=False)
cat_data = le.fit_transform(cat_data)

### Numerical data
# num_data = df[['Length(mm)','Angle']].values # Full Data
# num_data = df[['Length(mm)']].values  # Frag data only
num_data = df[['Angle']].values # Break data 
# num_data = df[[]] # No numerical vars

# Combine Categorical and numerical data
data = np.hstack((cat_data, num_data))
print(data.shape)

"""Now we apply a k-fold cross validation across multiple classifers."""

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
# Ignore all future convergence warnings (SVM might do this a lot)
simplefilter(action='ignore', category=ConvergenceWarning)

# Folds for K-Fold cross validation
reps = 300
folds = 10
cv = RepeatedKFold(n_splits = folds, n_repeats = reps)

### Random Forests
rf = RandomForestClassifier()
rf_acc = 100*cross_val_score(rf, data, labels, cv = cv, scoring = 'accuracy')
print(f'RF: {np.mean(rf_acc):.2f}% ± {(2.0 * np.std(rf_acc)):.2f}%')

### Support Vector Machines
clfsvm = SVC(kernel ="linear", max_iter = 5000)
scalar = StandardScaler() # SVM works best with standard scaled data
pipeline = Pipeline([('transformer', scalar), ('estimator', clfsvm)]) # That's what this does, in a safe manner
svm_acc = 100*cross_val_score(pipeline, data, labels, cv = cv, scoring = 'accuracy')
print(f'SVM: {np.mean(svm_acc):.2f}% ± {(2.0 * np.std(svm_acc)):.2f}%')

### LDA
clfLDA = LinearDiscriminantAnalysis()
lda_acc = 100*cross_val_score(clfLDA, data, labels, cv = cv, scoring='accuracy')
print(f'LDA: {np.mean(lda_acc):.2f}% ± {(2.0 * np.std(lda_acc)):.2f}%')

### Naive Bayes
clfgnb = GaussianNB()
gnb_acc = 100*cross_val_score(clfgnb, data, labels, cv = cv, scoring='accuracy')
print(f'GNB: {np.mean(gnb_acc):.2f}% ± {(2.0 * np.std(gnb_acc)):.2f}%')

### KNN
k = 25
clfknn= KNeighborsClassifier(n_neighbors= k)
knn_acc = 100*cross_val_score(clfknn, data, labels, cv = cv, scoring='accuracy')
print(f'KNN: {np.mean(knn_acc):.2f}% ± {(2.0 * np.std(knn_acc)):.2f}%')

### We group all the scores together, and then find the mean and the 2 * the standard error
# acc = np.vstack((rf_acc, svm_acc, lda_acc, gnb_acc, knn_acc))
# print(f'Mean Acu: {np.mean(acc):.2f}% ± {(2.0 * np.std(acc)):.2f}%')