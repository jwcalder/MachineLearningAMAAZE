import os
import glob
import pandas as pd
import numpy as np
import pickle

#The 

# Set directory if need be
#os.chdir(r"C:/home/amaaze/Documents/ExperimentalData)

#Get all the filenames
#filenames=[i for i in glob.glob("VirtualGoniometer_Measurements_*.csv")]

#combine all files in filenames
#combined_csv = pd.concat([pd.read_csv(f) for f in filenames ], ignore_index=True)

#export to csv
#combined_csv.to_csv( "allangles.csv", index=False, encoding='utf-8-sig')
df = pd.read_csv('allangles.csv')

#Sort data
df = df.sort_values(by=['Mesh Name','Break #'], ignore_index=True)

#Empty python dictionary
angledata = {}

#Loop over existing dataframe
start_index = 0
current_break = df['Break #'][0]
current_mesh = df['Mesh Name'][0]
for i in range(len(df)+1):

    #Check if we are moving on to a new break # or mesh, or we got to the end of the df
    if (i == len(df)) or (current_break != df['Break #'][i]) or (current_mesh != df['Mesh Name'][i]):

        #Gather all measurement data
        angles = df['Angle'].values[start_index:i].astype(float)
        num_vert = df['Number of Vertices'].values[start_index:i].astype(int)
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
        angledata[current_mesh][current_break] = {'Number of Measurements':len(angles),'Angle':angles, 'Number of Vertices':num_vert, 'Radius':radius, 'x':x, 'y':y, 'z':z, 'Fit':fit, 'Segmentation Parameter':segparam}

        #If we are not at the end, then setup for next min 
        if i < len(df):
            #Set new mesh and break #
            current_break = df['Break #'][i]
            current_mesh = df['Mesh Name'][i]

            #Set start index
            start_index = i


#Write dictionary to file
f = open('break_curve_data.pkl','wb')
pickle.dump(angledata,f)
f.close()


