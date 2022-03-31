#This is the script that is used to reorganize the data from the .pp files into a .csv format.
from utils import data_dir
import meshlab_pickedpoints as mpp
import pandas as pd
import numpy as np
import glob
import csv

outputFile = "ridge_endpoints.csv"

dir = data_dir + 'finaldata_ppfiles/'
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

structList = []
for i in range(len(fileList)):
    structList.append(pointsStruct(fileList[i], fragmentList[i]))

headerList = ["Specimen", "Point_Number", "x", "y", "z"]
with open(outputFile, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headerList)
    for struct in structList:
        writer.writerows(struct.collapse())

arr = np.empty((0,5), int)
for struct in structList:
    arr = np.append(arr, struct.collapse(), axis=0)
df = pd.DataFrame(arr, columns=["Specimen", "Point_Number", "x", "y", "z"])
