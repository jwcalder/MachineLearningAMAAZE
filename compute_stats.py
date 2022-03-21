import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import spatial

#Load angle data dictionary
with open('break_curve_data.pkl', 'rb') as f:
    angledata = pickle.load(f)

print('Specimen,Break #,Count,Mean,Median,STD,Max,Min')
#Loop over the breaks for meshname
for mesh_name in angledata:
    #print(mesh_name)
    for break_num in angledata[mesh_name]:
        #print('Break #:',break_num)
        print(mesh_name[:10],end=',')
        print(break_num,end=',')
        angles = angledata[mesh_name][break_num]['Angle']
        count = len(angles)
        mean = np.mean(angles)
        median = np.median(angles)
        std = np.std(angles)
        max_angle = np.max(angles)
        min_angle = np.min(angles)

        #Must go in same order as above
        print(count,end=',')
        print(mean,end=',')
        print(median,end=',')
        print(std,end=',')
        print(max_angle,end=',')
        print(min_angle)
        #x = angledata[mesh_name][break_num]['x']
        #y = angledata[mesh_name][break_num]['y']
        #z = angledata[mesh_name][break_num]['z']

 
