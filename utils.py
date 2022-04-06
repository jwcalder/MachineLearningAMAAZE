import numpy as np
import amaazetools.trimesh as tm
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt
from scipy import spatial
import os,sys
import pandas as pd


#Put all possible directory locations here
data_dirs = ['/drive/GoogleDrive/AMAAZE/Dissertation_YezziWoodley/Paper3_MoclanReplicationPaper/',
             '/Users/jeff/Moclan/Paper3_MoclanReplicationPaper/',
             '/data/ML_data/']

break_level_fields_numerical = ['Count',
                                'Mean',
                                'Median',
                                'STD',
                                'Max',
                                'Min',
                                'Range',
                                'ArcLength',
                                'EuclideanLength',
                                'ArcAngle',
                                'Surface Area',
                                'Volume',
                                'Bounding Box Dim1',
                                'Bounding Box Dim2',
                                'Bounding Box Dim3']

break_level_fields_categorial = ['interior_edge',
                                 'Interrupted',
                                 'ridge_notch',
                                 'interior_notch',
                                 'SzCl',
                                 'SkelPort',
                                 'LPort',
                                 'Element',
                                 'Side',
                                 'trab']
                                   
#Fields to compute summary statistics of
frag_level_sum_stats_fields = ['Mean',
                               'Median',
                               'STD',
                               'Max',
                               'Min',
                               'Range',
                               'ArcLength',
                               'EuclideanLength',
                               'ArcAngle']

#Break summary statistics to use at fragment level 
frag_level_sum_stats = ['min','max','mean','median','std']

#Frag level count fields
frag_level_count_fields =  ['interior_edge',
                            'Interrupted',
                            'ridge_notch',
                            'interior_notch']

#Frag level categorical fields
frag_level_fields_categorical = ['SzCl',
                                 'SizeRangeLb',
                                 'SizeRangeKg',
                                 'SkelPort',
                                 'LPort',
                                 'Element',
                                 'Side',
                                 'trab']

#Frag level numerical fields
frag_level_fields_numerical = ['Surface Area',
                               'Volume',
                               'Bounding Box Dim1',
                               'Bounding Box Dim2',
                               'Bounding Box Dim3']

#Find the directory that exists on a particular machine
try:
    data_dir_enum = enumerate(data_dirs)
    _,data_dir = next(data_dir_enum)
    while not os.path.isdir(data_dir):
        _,data_dir = next(data_dir_enum)
except:
    print('Warning: Could not find data directory.')

def break_level_ml_dataset(numerical_fields=None, categorical_fields=None, target_field='Effector',
                           standard_scaler=False):
    """Break Level Machine Learning Dataset
    ========

    Converts the break level data to numerical data via one-hot encodings and 
    returns a numerical dataset at the break-level for use in machine learning.
    
    Parameters
    ----------
    numerical_fields : list (optional)
        List of numerical fields to use. Uses all if not provided.
    categorical_fields : list (optional)
        List of categorical fields to use. Uses all if not provided.
    target_field : string (optional), default = 'Effector'
        Field to use for target of machine learning.
    standard_scaler : bool (optional), default = False
        Whether to apply standard scaling to the dataset.

    Returns
    -------
    data : numpy array (float)
        Features.
    target : numpy array (int)
        Targets as integers
    specimens : numpy array (string)
        List of specimen names.
    break_numbers : numpy array (int)
        Break number for each specimen.
    target_names : numpy array (string)
        List of target names.
        
    """

    if numerical_fields is None:
        numerical_fields = break_level_fields_numerical
    if categorical_fields is None:
        categorical_fields = break_level_fields_categorial

    df = pd.read_csv('break_level_ml.csv') 

    #Categorical Data
    cat_data = df[categorical_fields]
    le = preprocessing.OneHotEncoder(sparse=False)
    cat_data = le.fit_transform(cat_data)

    #Numerical data
    num_data = df[numerical_fields].values

    #Combine data
    data = np.hstack((cat_data,num_data))

    #Target
    le = preprocessing.LabelEncoder()
    target = le.fit_transform(df[target_field])
    target_names = le.classes_

    #Specimen names and break numbers
    specimens = df['Specimen'].values
    break_numbers = df['BreakNo'].values

    #Standard scaler
    if standard_scaler:
        data = preprocessing.StandardScaler().fit_transform(data)  # Scaling data

    return data,target,specimens,break_numbers,target_names

def frag_level_ml_dataset(numerical_fields=None, categorical_fields=None, sum_stats_fields=None,
                          sum_stats=None, count_fields=None, target_field='Effector', standard_scaler=False):
    """Fragment Level Machine Learning Dataset
    ========

    Converts the fragment level data to numerical data via one-hot encodings and 
    returns a numerical dataset at the fragment-level for use in machine learning.
    
    Parameters
    ----------
    numerical_fields : list (optional)
        List of numerical fields to use. Uses all if not provided.
    categorical_fields : list (optional)
        List of categorical fields to use. Uses all if not provided.
    sum_stats_fields : list (optional)
        List of summary statistics fields to use. Uses all if not provided.
    sum_stats : list (optional)
        Summary statistics to use. Uses all if not provided.
    count_fields : list (optional)
        List of categorical counting fields.
    target_field : string (optional), default = 'Effector'
        Field to use for target of machine learning.
    standard_scaler : bool (optional), default = False
        Whether to apply standard scaling to the dataset.

    Returns
    -------
    data : numpy array (float)
        Features.
    target : numpy array (int)
        Targets as integers
    specimens : numpy array (string)
        List of specimen names.
    target_names : numpy array (string)
        List of target names.
    """

    if numerical_fields is None:
        numerical_fields = frag_level_fields_numerical
    if categorical_fields is None:
        categorical_fields = frag_level_fields_categorical
    if sum_stats_fields is None:
        sum_stats_fields = frag_level_sum_stats_fields
    if sum_stats is None:
        sum_stats = frag_level_sum_stats
    if count_fields is None:
        count_fields = frag_level_count_fields

    df = pd.read_csv('frag_level_ml.csv') 

    #Categorical Data
    cat_data = df[categorical_fields]
    le = preprocessing.OneHotEncoder(sparse=False)
    cat_data = le.fit_transform(cat_data)

    #Add all summary statistics fields
    for field in sum_stats_fields:
        for stat in sum_stats:
            numerical_fields += [field + '_' + stat]

    #Add all count fields
    for field in count_fields:
        for c in df.columns:
            if c.startswith(field):
                numerical_fields += [c]
                    
    #Numerical data
    num_data = df[numerical_fields].values

    #Combine data
    data = np.hstack((cat_data,num_data))

    #Target
    le = preprocessing.LabelEncoder()
    target = le.fit_transform(df[target_field])
    target_names = le.classes_

    #Specimen names and break numbers
    specimens = df['Specimen'].values

    #Standard scaler
    if standard_scaler:
        data = preprocessing.StandardScaler().fit_transform(data)  # Scaling data

    return data,target,specimens,target_names

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
    df = pd.read_csv(data_dir + 'finaldata_inventoryall.csv', encoding = 'cp1252')
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

def arc_length(x,y,z):
    """Arc Length
    ========

    Computes the arclength of a path.
    
    Parameters
    ----------
    x : numpy array
        x-coordinates along path.
    y : numpy array
        y-coordinates along path.
    z : numpy array
        z-coordinates along path.

    Returns
    -------
    length : float
        Arclength of path.
    """
    v = np.vstack((x,y,z))
    length = 0
    for i in range(len(x)-1):
        length += np.linalg.norm(v[:,i+1] - v[:,i])
    return length

def arc_angle(x,y,z,principal_dir):
    """Arc Angle
    ========

    Computes the arcangle of a break relative to principal axis of fragment.
    
    Parameters
    ----------
    x : numpy array
        x-coordinates along break.
    y : numpy array
        y-coordinates along break.
    y : numpy array
        y-coordinates along break.
    principal_dir : numpy array
        Principal axis of the fragment.

    Returns
    -------
    angle : float
        Arcangle
    """
    v = np.vstack((x,y,z)).T
    _, vecs = tm.pca(v)
    break_dir = vecs[:,0]
    angle = np.arccos(np.abs(np.dot(principal_dir,break_dir)))*180/np.pi
    return angle
        
def tsp_order(x,y,z):
    """Travelling SalesPerson (TSP) Ordering
    ========

    Returns an ordering of the points along a path that minimizes the distance travelled,
    which is the travelling salesperson (TSP) problem.
    
    Parameters
    ----------
    x : numpy array
        x-coordinates along path.
    y : numpy array
        y-coordinates along path.
    y : numpy array
        y-coordinates along path.

    Returns
    -------
    min_path : list
        Indices of TSP ordering.
    """
    n = len(x)
    X = np.stack((x,y,z)).T
    D = spatial.distance_matrix(X,X)
    D[range(n),range(n)] = np.inf

    #Try starting from each point
    min_length = np.ones((n-1,))*np.inf
    min_path = list(range(n))
    for i in range(n):
        path,length = tsp_path(D.copy(),i)
        if np.sum(length) < np.sum(min_length):
            min_length = length
            min_path = path
    return min_path

def tsp_path(D,i):
    n = D.shape[0]
    path = [i]
    length = []
    for j in range(n-1):
        path += [np.argmin(D[path[j],:])]
        length += [D[path[j],path[j+1]]]
        D[:,path[j]] = np.inf
    
    return path,np.array(length)


def save_curve(x,y,z,idx,break_num,fname):

    fig = plt.figure(figsize=(10,10))
    fig.suptitle('Break %d'%break_num)
    ax = fig.add_subplot(2,2,1,projection='3d')
    ax.scatter(x[idx[1:-1]],y[idx[1:-1]],z[idx[1:-1]],c='b')
    ax.scatter(x[idx[0]],y[idx[0]],z[idx[0]],c='r',s=100)
    ax.scatter(x[idx[-1]],y[idx[-1]],z[idx[-1]],c='g',s=100)
    ax.plot(x[idx],y[idx],z[idx])
    ax.set_title('(0,0)')
    ax.view_init(elev=0., azim=0)

    ax = fig.add_subplot(2,2,2,projection='3d')
    ax.scatter(x[idx[1:-1]],y[idx[1:-1]],z[idx[1:-1]],c='b')
    ax.scatter(x[idx[0]],y[idx[0]],z[idx[0]],c='r',s=100)
    ax.scatter(x[idx[-1]],y[idx[-1]],z[idx[-1]],c='g',s=100)
    ax.plot(x[idx],y[idx],z[idx])
    ax.set_title('(0,90)')
    ax.view_init(elev=0., azim=90)

    ax = fig.add_subplot(2,2,3,projection='3d')
    ax.scatter(x[idx[1:-1]],y[idx[1:-1]],z[idx[1:-1]],c='b')
    ax.scatter(x[idx[0]],y[idx[0]],z[idx[0]],c='r',s=100)
    ax.scatter(x[idx[-1]],y[idx[-1]],z[idx[-1]],c='g',s=100)
    ax.plot(x[idx],y[idx],z[idx])
    ax.set_title('(45,0)')
    ax.view_init(elev=45., azim=0)

    ax = fig.add_subplot(2,2,4,projection='3d')
    ax.scatter(x[idx[1:-1]],y[idx[1:-1]],z[idx[1:-1]],c='b')
    ax.scatter(x[idx[0]],y[idx[0]],z[idx[0]],c='r',s=100)
    ax.scatter(x[idx[-1]],y[idx[-1]],z[idx[-1]],c='g',s=100)
    ax.plot(x[idx],y[idx],z[idx])
    ax.set_title('(90,0)')
    ax.view_init(elev=90., azim=0)

    plt.savefig(fname)
    plt.close()

def plot_curve(x,y,z,idx,break_num):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[idx[1:-1]],y[idx[1:-1]],z[idx[1:-1]],c='b')
    ax.scatter(x[idx[0]],y[idx[0]],z[idx[0]],c='r',s=100)
    ax.scatter(x[idx[-1]],y[idx[-1]],z[idx[-1]],c='g',s=100)
    ax.plot(x[idx],y[idx],z[idx])
    ax.set_title('Break %d'%break_num)


