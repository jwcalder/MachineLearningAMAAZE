from utils import data_dir
import pickle
import numpy as np
import pandas as pd

#Fields to copy
fields = ['Species', 
          'Common', 
          'SzCl', 
          'SizeRangeLb', 
          'SizeRangeKg', 
          'SkelPort', 
          'LPort', 
          'Element', 
          'Side', 
          'ActorTaxon', 
          'Effector']

df = pd.read_csv(data_dir + 'finaldata_inventoryall.csv', encoding = 'cp1252')
df = df[['Specimen'] + fields]

#Load angle data dictionary
with open('break_curve_data.pkl', 'rb') as f:
    angledata = pickle.load(f)

specimen_list = []
for specimen in angledata:
    specimen_list += [specimen]

keep = np.zeros(len(df),dtype=bool)
for i in range(len(df)):
    keep[i] = df['Specimen'][i] in specimen_list

df = df[keep]
df = df.to_csv('inventory_data.csv', index=False)


