from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, RepeatedKFold
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ssl
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

ssl._create_default_https_context = ssl._create_unverified_context

#df = pd.read_csv('moclan.csv')
df = pd.read_csv('moclan_carngrouped.csv')

### Bootstrap
# boot_reps = 10000 # How many copies we want
# boot_strap = df.sample(boot_reps, replace=True)
# df = pd.concat([df, boot_strap])

### Labels
le = preprocessing.LabelEncoder()
labels = le.fit_transform(df['Agent'])

### Categorical Data
# Note that Epiphysis, Notch_a, and Notch_c all contain levels beyond "Absent" and "Present"
# This is unexpected for a boolean variable. Those variables are ommitted in the "clean" vector. 

# cat_data = df[['Epiphysis', 'Number_of_planes', 'Type_of_angle', 'Notch', 'Notch_a', 'Notch_c', 'Notch_d', 'Fracture_plane', '>4cm', 'Interval(length)']] # Full data
# cat_data = df[['Number_of_planes', 'Type_of_angle', 'Notch', 'Notch_d', 'Fracture_plane', '>4cm', 'Interval(length)']] # Full data without unclean data
# cat_data = df[['Epiphysis', 'Number_of_planes','Notch','Notch_a','Notch_c','Notch_d','Interval(length)']] # Frag only
cat_data = df[['Type_of_angle', 'Fracture_plane', '>4cm']] # Break level only
# cat_data = df[[]]  #no cat vars

le = preprocessing.OneHotEncoder(sparse=False)
cat_data = le.fit_transform(cat_data)

### Numerical data
# num_data = df[['Length(mm)','Angle','Number_of_planes']].values # Full Data
#num_data = df[['Length(mm)']].values  # Frag data only
num_data = df[['Angle']].values # Break data only

# Combine Categorical and numerical data
data = np.hstack((cat_data, num_data))
print(data.shape)

"""Now we do a train test split and apply some machine learning."""

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
# Ignore all future convergence warnings (SVM might do this a lot)
simplefilter(action='ignore', category=ConvergenceWarning)

# Folds for K-Fold cross validation
reps = 100
folds = 3
cv = RepeatedKFold(n_splits = folds, n_repeats = reps)

### Random Forests
rf = RandomForestClassifier()
rf_acc = 100*cross_val_score(rf, data, labels, cv = cv, scoring = 'accuracy')
print(f'RF: {np.mean(rf_acc):.2f}% ± {(2.0 * np.std(rf_acc)):.2f}%')

### Support Vector Machines
clfsvm = SVC(kernel ="linear", max_iter = 2000)
scalar = StandardScaler() # SVM works best with standard scaled data
pipeline = Pipeline([('transformer', scalar), ('estimator', clfsvm)]) # That's what this does, in a safe manner
svm_acc = 100*cross_val_score(pipeline, data, labels, cv = cv, scoring = 'accuracy')
print(f'SVM: {np.mean(svm_acc):.2f}% ± {(2.0 * np.std(svm_acc)):.2f}%')

### Decision Trees
clfTree = DecisionTreeClassifier()
tree_acc = 100*cross_val_score(clfTree, data, labels, cv = cv, scoring='accuracy')
print(f'DT: {np.mean(tree_acc):.2f}% ± {(2.0 * np.std(tree_acc)):.2f}%')

### We group all the scores together, and then find the mean and the 2 * the standard error
acc = np.vstack((rf_acc, svm_acc, tree_acc))
print(f'Mean Acu: {np.mean(acc):.2f}% ± {(2.0 * np.std(acc)):.2f}%')