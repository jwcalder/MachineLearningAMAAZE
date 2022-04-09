import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 12})
styles = ['^-', 'o-', 'd-', 's-', 'p-', 'x-', '*-']
data = pd.read_csv('results/gbl_ml_accuracies_angular.csv')
x = np.array([data.iloc[:,0],data.iloc[:,501]])
plt.plot(x[0]*100,x[1], marker = 'o', label = 'Angular Weight Matrix')
data = pd.read_csv('results/gbl_ml_accuracies_euclidean.csv')
x = np.array([data.iloc[:,0],data.iloc[:,501]])
plt.plot(x[0]*100,x[1], marker = 'D', label = 'Euclidean Weight Matrix')
plt.xlabel('Percentage Training Data')
plt.ylabel('Average Accuracy (Percentage)')
plt.title('Average accuracy of GBL VS. percentage training data')
plt.legend()
plt.savefig('figures/gbl_accuracy_vs_training_percentage')

