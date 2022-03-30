import numpy as np
import pickle
import pandas as pd
import math

def arc_distance(mini_dict):
    sum_dist = 0
    num_measurements = len(mini_dict['x'])
    if num_measurements == 1:
        print("oops") # Also, this will crash. 
        
    for i in range(1,num_measurements):
        # We find the distance between each point and the next one.
        # This is just the distance between two points formula applied to 3 dimensions. 
        x1 = mini_dict['x'][i-1]
        x2 = mini_dict['x'][i]
        xdist = (x2-x1)**2
        
        y1 = mini_dict['y'][i-1]
        y2 = mini_dict['y'][i]
        ydist = (y2 - y1)**2
        
        z1 = mini_dict['z'][i-1]
        z2 = mini_dict['z'][i]
        zdist = (z2 - z1)**2
        
        # And then we add all of it up. 
        sum_dist += math.sqrt(xdist+ydist+zdist) 
    
    return sum_dist

def arc_angle(mini_dict, boundingBoxPlane):
    num_measurements = len(mini_dict['x']) 
    # This won't crash if there's only one measurement, but the angle WILL be garbage.
    
    x = np.zeros(num_measurements)
    y = np.zeros(num_measurements)
    z = np.zeros(num_measurements)
    for i in range(num_measurements): 
        x[i] = mini_dict['x'][i]
        y[i] = mini_dict['y'][i]
        z[i] = mini_dict['z'][i]
        
    data = np.concatenate((x[:, np.newaxis],
                           y[:, np.newaxis],
                           z[:, np.newaxis]),
                          axis = 1)
    
    datamean = data.mean(axis = 0)
    
    _, _, vv = np.linalg.svd(data - datamean)
    
    line = vv[0] # This is our line of best fit
    
    norm = boundingBoxPlane / np.linalg.norm(boundingBoxPlane)
    line = line / np.linalg.norm(line)
    
    dot = np.dot(norm, line)
    dot = np.median([-1, dot, 1]) # Contain the value to the domain of arcsin
    
    angle = math.degrees(np.arcsin(dot)) # Radians are harder to interpret
    
    # We take the absolute value of the angle because the vector 'direction'
    # Is arbitrary. This doens't effect the magnitude of the angle, just the sign.
    
    return abs(angle) # This is the angle we want!

arc_df = pd.DataFrame()
arc_df['Mesh_Name'] = []
arc_df['BreakNo'] = [] 
arc_df['Arc_Length'] = []
arc_df['Arc_Angle'] = []

with open('break_curve_data.pkl', 'rb') as f:
    angledata = pickle.load(f)
    
df = pd.read_csv('mesh_stats_final.csv', encoding = 'cp1252')

for mesh_name in angledata:
    # print(mesh_name)
    mesh = df.loc[df['Specimen'] == mesh_name]
    for break_num in angledata[mesh_name]:
        # print('Break #:',break_num)
        normals = [mesh['nv a 1'].iloc[0], mesh['nv a 2'].iloc[0], mesh['nv a 3'].iloc[0]]
        length = arc_distance(angledata[mesh_name][break_num])
        angle = arc_angle(angledata[mesh_name][break_num], normals)
        arc_df.loc[len(arc_df)] = [mesh_name,
                                   break_num,
                                   length,
                                   angle]
        
arc_df.to_csv('final_arcLA.csv',index=False)  
