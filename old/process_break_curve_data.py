import os
import glob
import pandas as pd
import numpy as np
import pickle

AMAAZE ='/drive/GoogleDrive/AMAAZE/'
file = AMAAZE + 'Dissertation_YezziWoodley/Paper3_MoclanReplicationPaper/finaldata_angle_level.csv'

df = pd.read_csv(file)

#Sort data
df = df.sort_values(by=['Mesh_Name','BreakNo'], ignore_index=True)

#Empty python dictionary
angledata = {}

#Loop over existing dataframe
start_index = 0
current_break = df['BreakNo'][0]
current_mesh = df['Mesh_Name'][0]
for i in range(len(df)+1):

    #Check if we are moving on to a new break # or mesh, or we got to the end of the df
    if (i == len(df)) or (current_break != df['BreakNo'][i]) or (current_mesh != df['Mesh_Name'][i]):

        #Gather all measurement data
        angles = df['Angle'].values[start_index:i].astype(float)
        num_vert = df['Number_of_Vertices'].values[start_index:i].astype(int)
        radius = df['Radius'].values[start_index:i].astype(float)
        x = df['x'].values[start_index:i].astype(float)
        y = df['y'].values[start_index:i].astype(float)
        z = df['z'].values[start_index:i].astype(float)
        fit = df['fit'].values[start_index:i].astype(float)
        segparam = df['SegParam'].values[start_index:i].astype(float)

        #Initialize dictionary if not already initialized
        if current_mesh not in angledata.keys():
            angledata[current_mesh] = {}

        #add break curve to dictionary for current mesh name
        #Add NotesElement
        angledata[current_mesh][current_break] = {'Number_of_Vertices':len(angles),'Angle':angles, 'Number_of_Vertices':num_vert, 'Radius':radius, 'x':x, 'y':y, 'z':z, 'Fit':fit, 'Segmentation Parameter':segparam}

        #If we are not at the end, then setup for next min 
        if i < len(df):
            #Set new mesh and break #
            current_break = df['BreakNo'][i]
            current_mesh = df['Mesh_Name'][i]

            #Set start index
            start_index = i


#Write dictionary to file
f = open('break_curve_data.pkl','wb')
pickle.dump(angledata,f)
f.close()


