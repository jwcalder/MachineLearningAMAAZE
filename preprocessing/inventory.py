import pandas as pd
import pickle
import numpy as np

def sample_inventory(fields):
    """Sample Inventory
    ========

    Pulls data from the main inventory file corresponding to our sample.
    Always includes the number of breaks for each fragment as well.
    
    Parameters
    ----------
    fields : list 
        List of fields from main inventory file to pull.

    Returns
    -------
    df : panads datafram
        Returns a pandas dataframe. Always includes the number of breaks for each fragment as well.
    """

    #Load inventory file
    df = pd.read_csv('finaldata_inventoryall.csv', encoding = 'cp1252')
    df = df[fields]

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

    num_breaks = np.zeros(len(df))
    for i in range(len(df)):
        num_breaks[i] = len(angledata[df['Specimen'][i]])
    df['NumBreaks'] = num_breaks.astype(int)

    return df

