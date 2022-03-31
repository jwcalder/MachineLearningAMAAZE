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
df = df[keep].reset_index().drop(['index'],axis=1)

#Add trabecular data as well
trab_df = pd.read_csv(data_dir+'finaldata_trabecula.csv', encoding = 'cp1252')
trab = []
for i in range(len(df)):
    trab_row = trab_df.loc[trab_df['Specimen'] == df['Specimen'][i]]
    trab += [str(trab_row['trab'].iloc[0])]
df['trab'] = trab

#Add mesh_stats.csv to this
mesh_stats_df = pd.read_csv('mesh_stats.csv', encoding = 'cp1252')
mesh_stats_fields = ['Surface Area', 'Volume', 'Bounding Box Dim1', 'Bounding Box Dim2', 'Bounding Box Dim3']

for field in mesh_stats_fields:
    vals = []
    for i in range(len(df)):
        mesh_stats_row = mesh_stats_df.loc[mesh_stats_df['Specimen'] == df['Specimen'][i]]
        vals += [mesh_stats_row[field].iloc[0]]
    df[field] = vals

df.to_csv('frag_level_data.csv', index=False)
