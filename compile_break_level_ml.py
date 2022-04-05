import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from utils import arc_length, arc_angle, data_dir
from scipy import spatial
import sys

#Load angle data dictionary
with open('break_curve_data.pkl', 'rb') as f:
    angledata = pickle.load(f)

#Open mesh_stats data frame
mesh_stats_df = pd.read_csv('mesh_stats.csv', encoding = 'cp1252')

#Open manual data
manual_data_df = pd.read_csv(data_dir+'manual_break_level.csv', encoding = 'cp1252')
manual_fields = ['interior_edge','Interrupted','Interruptedby','ridge_notch','interior_notch']
frag_df = pd.read_csv('frag_data.csv')

#Frag level fields
frag_fields = ['Species','Common','SzCl','SizeRangeLb','SizeRangeKg','SkelPort','LPort','Element','Side','ActorTaxon','Effector','trab','Surface Area','Volume','Bounding Box Dim1','Bounding Box Dim2','Bounding Box Dim3']

sys.stdout = open('break_level_ml.csv', 'w')
print('Specimen,BreakNo,Count,Mean,Median,STD,Max,Min,Range,ArcLength,EuclideanLength,ArcAngle',end='')
for field in manual_fields:
    print(','+field,end='')
for field in frag_fields:
    print(','+field,end='')
print('')
#Loop over the meshes
for mesh_name in angledata:
    
    mesh_stats = mesh_stats_df.loc[mesh_stats_df['Specimen'] == mesh_name]
    manual_data = manual_data_df.loc[manual_data_df['Specimen'] == mesh_name]
    frag_data = frag_df.loc[frag_df['Specimen'] == mesh_name]

    for break_num in angledata[mesh_name]:
        print(mesh_name[:10],end=',')
        print(break_num,end=',')
        angles = angledata[mesh_name][break_num]['Angle']
        count = len(angles)
        mean = np.mean(angles)
        median = np.median(angles)
        std = np.std(angles)
        max_angle = np.max(angles)
        min_angle = np.min(angles)
        angle_range = max_angle - min_angle

        #Get x,y,z coordinates along arc
        x = angledata[mesh_name][break_num]['x']
        y = angledata[mesh_name][break_num]['y']
        z = angledata[mesh_name][break_num]['z']

        #Arc length
        al = arc_length(x,y,z)

        #Euclidean length
        el = np.sqrt( (x[0] - x[-1])**2 + (y[0] - y[-1])**2 + (z[0] - z[-1])**2)

        #Arc angle
        principle_dir = np.array([mesh_stats['nv a 1'].iloc[0], mesh_stats['nv a 2'].iloc[0], mesh_stats['nv a 3'].iloc[0]])
        aa = arc_angle(x,y,z,principle_dir)

        #Must go in same order as above
        print(count,end=',')
        print(mean,end=',')
        print(median,end=',')
        print(std,end=',')
        print(max_angle,end=',')
        print(min_angle,end=',')
        print(angle_range,end=',')
        print(al,end=',')
        print(el,end=',')
        print(aa,end='')

        for field in manual_fields:
            print(','+str(manual_data[field].iloc[0]),end='')

        for field in frag_fields:
            print(','+str(frag_data[field].iloc[0]),end='')

        print('')

sys.stdout.close()
 
