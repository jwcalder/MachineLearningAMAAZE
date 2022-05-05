import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 12})
styles = ['^-', 'o-', 'd-', 's-', 'p-', 'x-', '*-']

#data = pd.read_csv('results/gbl_ml_accuracies_angular.csv')
#x = np.array([data.iloc[:,0],data.iloc[:,501]])
#plt.plot(x[0]*100,x[1], marker = 'o', label = 'Angular Weight Matrix')

data = pd.read_csv('results/gbl_regular_info.csv')
x = np.array([data.iloc[:,0],data.iloc[:,1]])
plt.plot(x[0],x[1], marker = 'D', label = 'Graph-based Learning')

data = pd.read_csv('results/rf_info.csv')
x = np.array([data.iloc[:,0], data.iloc[:,1]])
plt.plot(x[0],x[1],marker = 's', label = 'Random Forest')

data = pd.read_csv('results/gbl_spectral_info.csv')
x = np.array([data.iloc[:,0], data.iloc[:,1]])
plt.plot(x[0],x[1],marker = 's', label = 'Graph-based Learning, with SE')

data = pd.read_csv('results/gbl_vae_info.csv')
x = np.array([data.iloc[:,0], data.iloc[:,1]])
plt.plot(x[0],x[1],marker = 's', label = 'Graph-based Learning, with VAE')

plt.xlabel('Percentage Training Data')
plt.ylabel('Average Accuracy (Percentage)')
plt.title('Average accuracy VS. percentage training data')
plt.ylim(60,100)
plt.legend()
plt.savefig('figures/gbl_accuracy_vs_training_percentage')

