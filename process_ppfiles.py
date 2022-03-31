#This is the script that is used to reorganize the data from the .pp files into a .csv format.
from utils import data_dir
import meshlab_pickedpoints as mpp
import pandas as pd
import numpy as np
import glob
import csv

outputFile = "ridge_endpoints.csv"

dir = 'ppfiles/'
fileList = [f for f in glob.glob(dir + "*.pp")]

fragmentList = []
for file in fileList:
    fragmentList.append(mpp.load(file))

class pointsStruct:
    def __init__(self, name, points):
        self.name = name
        self.points = points
    
    def collapse(self):
        out = []
        for line in self.points:
            inner = [self.name[len(dir):len(dir)+10], int(line["name"])]
            inner.extend(line["point"])
            out.append(inner)
        return out

arr = np.empty((0,5), int)
for i in range(len(fileList)):
    arr = np.append(arr, pointsStruct(fileList[i], fragmentList[i]).collapse(), axis=0)
pp_df = pd.DataFrame(arr, columns=["Specimen", "Point_Number", "x", "y", "z"])

#Convert to a dictionary
pp_dict = {}
for i in range(len(pp_df)):

    specimen = pp_df['Specimen'][i]
    point_num = int(pp_df['Point_Number'][i])
    x,y,z = pp_df['x'][i],pp_df['y'][i],pp_df['z'][i]

    #Initialize dictionary if not already initialized
    if specimen not in pp_dict.keys():
        pp_dict[specimen] = {}

    pp_dict[specimen][point_num] = [x,y,z]

df = pd.read_csv(data_dir + "manual_break_level.csv", usecols = ['Specimen', 'ep1', 'ep2'])
df["ep1_x"] = 0 
df["ep1_y"] = 0
df["ep1_z"] = 0

df["ep2_x"] = 0
df["ep2_y"] = 0
df["ep2_z"] = 0

for i in range(len(df)):
    specimen = df['Specimen'][i]
    ep1 = df['ep1'][i]
    ep2 = df['ep2'][i]

    df.loc[i, "ep1_x"] = pp_dict[specimen][ep1][0]
    df.loc[i, "ep1_y"] = pp_dict[specimen][ep1][1]
    df.loc[i, "ep1_z"] = pp_dict[specimen][ep1][2]

    df.loc[i, "ep2_x"] = pp_dict[specimen][ep2][0]
    df.loc[i, "ep2_y"] = pp_dict[specimen][ep2][1]
    df.loc[i, "ep2_z"] = pp_dict[specimen][ep2][2]

df.to_csv('break_ep_data.csv', index=False)



