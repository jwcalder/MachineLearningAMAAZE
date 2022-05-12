from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import break_level_ml_dataset
from utils import frag_level_ml_dataset
from sklearn import preprocessing
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 12})
styles = ['^-', 'o-', 'd-', 's-', 'p-', 'x-', '*-']

data,target,specimens,break_numbers,target_names = frag_level_ml_dataset()
data = preprocessing.StandardScaler().fit_transform(data)

kmeans = KMeans(n_clusters = 3).fit(data)
sys.stdout = open('results/k_means_clusters_frag)', 'w')

for i in range(data.size):
    print(specimens[i], end = ', ')
    print(kmeans.labels_[i])
sys.close()