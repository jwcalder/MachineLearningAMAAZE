import pandas as pd
import numpy as np
import pickle
from utils import data_dir, tsp_order, save_curve
from collections import defaultdict

# Read in the main data
df = pd.read_csv(data_dir + 'finaldata_angle_level.csv', encoding = 'cp1252')
endpoints_df = pd.read_csv('break_ep_data.csv')

#Sort data
df = df.sort_values(by=['Mesh_Name','BreakNo'], ignore_index=True)
endpoints_df = endpoints_df.sort_values(by=['Specimen', 'BreakNo'], ignore_index=True)

# This generates a dictionary of all the meshes, their breaks, and those end points
endpoints_dict = defaultdict(dict)
for i in range(len(endpoints_df)):
    current_mesh = endpoints_df['Specimen'][i]
    current_break = endpoints_df['BreakNo'][i]
    ep1 = [endpoints_df['ep1_x'][i], endpoints_df['ep1_y'][i], endpoints_df['ep1_z'][i]]
    ep2 = [endpoints_df['ep2_x'][i], endpoints_df['ep2_y'][i], endpoints_df['ep2_z'][i]]
    endpoints_dict[current_mesh].update({current_break: [ep1, ep2]})
    
#Empty python dictionary
angledata = {}

#Loop over existing dataframe
start_index = 0
current_break = df['BreakNo'][0]
current_mesh = df['Mesh_Name'][0][:10]
for i in range(len(df)+1):

    #Check if we are moving on to a new break # or mesh, or we got to the end of the df
    if (i == len(df)) or (current_break != df['BreakNo'][i]) or (current_mesh != df['Mesh_Name'][i][:10]):
        
        # Get end points of current break
        ep1, ep2 = endpoints_dict[current_mesh][current_break]
        x1, y1, z1 = ep1
        x2, y2, z2 = ep2
        
        angles = df['Angle'].values[start_index:i].astype(float)
        num_vert = df['Number_of_Vertices'].values[start_index:i].astype(int)
        radius = df['Radius'].values[start_index:i].astype(float)
        
        # Because of what we're doing here, the xs, ys, and zs are going to be out of order. 
        x = df['x'].values[start_index:i].astype(float)
        x = np.hstack([x1, x, x2])
        
        y = df['y'].values[start_index:i].astype(float)
        y = np.hstack([y1, y, y2])
        
        z = df['z'].values[start_index:i].astype(float)
        z = np.hstack([z1, z, z2])
        
        #Order the points correctly along each curve
        idx = tsp_order(x,y,z)
        #save_curve(x,y,z,idx,current_break,'figures/tsp_'+current_mesh+'_'+str(current_break)+'.png')
        x,y,z = x[idx],y[idx],z[idx]

        fit = df['fit'].values[start_index:i].astype(float)
        segparam = df['SegParam'].values[start_index:i].astype(float)

        #Initialize dictionary if not already initialized
        if current_mesh not in angledata.keys():
            angledata[current_mesh] = {}

        #add break curve to dictionary for current mesh name
        angledata[current_mesh][current_break] = {'Number of Measurements':len(angles),'Angle':angles, 'Number of Vertices':num_vert, 'Radius':radius, 'x':x, 'y':y, 'z':z, 'Fit':fit, 'Segmentation Parameter':segparam}

        #If we are not at the end, then setup for next min 
        if i < len(df):
            #Set new mesh and break #
            current_break = df['BreakNo'][i]
            current_mesh = df['Mesh_Name'][i][:10]

            #Set start index
            start_index = i

#Write dictionary to file
f = open('break_curve_data.pkl','wb')
pickle.dump(angledata,f)
f.close()


