import amaazetools.trimesh as tm
from utils import data_dir
import pandas as pd
import numpy as np
import os
from tqdm import trange

#New dataframe
df = pd.DataFrame()
df['Specimen'] = []
df['Surface Area'] = [] 
df['Volume'] = []
df['Bounding Box Dim1'] = []
df['Bounding Box Dim2'] = []
df['Bounding Box Dim3'] = []
df['nv a 1'] = []
df['nv a 2'] = []
df['nv a 3'] = []
df['nv b 1'] = []
df['nv b 2'] = []
df['nv b 3'] = []
df['nv c 1'] = []
df['nv c 2'] = []
df['nv c 3'] = []

#Directory to look for ply files in
directory = data_dir + 'finaldata_VtgonMeshes'
files = os.listdir(directory)
for i in trange(len(files)): # If you've got a lot of files, this may take a while
    filename = files[i]
    if filename.endswith(".ply"):
      mesh = tm.load_ply(os.path.join(directory,filename))
      
      # The bounding box itself
      bbox = mesh.bbox()
      
      # Bounding box direction vectors
      _, vecs = tm.pca(mesh.points)
      
      # V1 is the longitudinal plane, I believe. 
      v1 = vecs[:,0] # I'm grabbing all of the vectors in case I'm using the wrong one
      v2 = vecs[:,1] # Because this script takes forever to run
      v3 = vecs[:,2] # I think it's because these ply files are pretty big
      
      df.loc[len(df)] = [filename[:10], 
                         mesh.surf_area(),
                         mesh.volume(), 
                         bbox[0], 
                         bbox[1], 
                         bbox[2],
                         v1[0],
                         v1[1],
                         v1[2],
                         v2[0],
                         v2[1],
                         v2[2],
                         v3[0],
                         v3[1],
                         v3[2]
                         ]

df.to_csv('mesh_stats.csv',index=False)      
