import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
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
             '/data/ML_data/',
             '/home/jeff/Dropbox/Work/AMAAZE/csv_files/']

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


class Net(nn.Module):
    def __init__(self, structure=[10,10], num_classes=2, dropout_rate=0.5, batch_normalization=True):
        """Neural Network Classifier
        ========

        General class for a neural network classifier in pytorch.
        
        Parameters
        ----------
        structure : list (optional)
            List giving number of inputs to each layer. 
            The default structure=[10,10,2] constructs a 2-layer neural network with 10 inputs 
             and 10 nerons in the first layer.
        num_classes : int (optional)
            Number of classes.
        dropout_rate : float (optional), default=0.5
            Dropout rate.
        batch_normalization : bool (optional), default=True
            Whether to apply batch normalization.
        """

        super(Net, self).__init__()

        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate
        self.num_layers = len(structure)-1
        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(self.num_layers):
            self.fc.append(nn.Linear(structure[i],structure[i+1]))
            self.bn.append(nn.BatchNorm1d(structure[i]))
        self.final = nn.Linear(structure[self.num_layers],num_classes)

        self.dropout = nn.Dropout(dropout_rate)

    def reset(self):

        self.final.reset_parameters()
        for i,l in enumerate(self.fc):
            l.reset_parameters()
            if self.batch_normalization:
                self.bn[i].reset_parameters()

    def forward(self, x):

        for i,l in enumerate(self.fc):
            if self.batch_normalization:
                x = self.bn[i](x)
            if self.dropout_rate > 0:
                x = self.dropout(x)
            x = F.relu(l(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)
        x = self.final(x)
        output = F.log_softmax(x, dim=1)
        return output

    def fit(self, data, target, epochs=100, cuda=True, learning_rate=0.1, 
            batch_size=32, gamma=0.9, verbose=False):
        """Fit
        ========

        Trains the neural network.
        
        Parameters
        ----------
        data : numpy array (float)
            Features
        target : numpy array (int)
            Labels.
        epochs : int (optional), default=100
            Number of training epochs (loops over whole dataset).
        cuda : bool (optional), default=True
            Whether to use GPU, if found.
        learning_rate : float (optional), defualt = 0.1
            Learning rate for training.
        batch_size : int (optional), default = 32
            Size of mini-batches for stochastic gradient descent.
        gamma : float (optional), default = 0.9
            Scheduler parameter. How much to decrease learning rate at each iteration.
        verbose : bool (optional), default = False
            Whether to print out details during training or not.
        """

        #Reset weights
        self.reset()

        #Convert to torch
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).long()

        #Cuda (GPU)
        use_cuda = cuda and torch.cuda.is_available()
        if verbose:
            if use_cuda:
                print("Using GPU!")
            else:
                print("Using CPU.")
        device = torch.device("cuda" if use_cuda else "cpu")
        self.to(device)

        #Optimizer and scheduler
        optimizer = optim.Adadelta(self.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        for epoch in range(1, epochs + 1):
            if verbose:
                print('\nEpoch: %d'%epoch)
            self.train_epoch(device, data, target, optimizer, epoch, batch_size, verbose)
            #test_acc = test(model, device, data_test, target_test, 'Test ')
            #train_acc = test(model, device, data_train, target_train, 'Train')
            scheduler.step()

    def train_epoch(self, device, data, target, optimizer, epoch, batch_size, verbose):

        self.train()
        batch_idx = 0
        for idx in range(0,len(data),batch_size):
            data_batch, target_batch = data[idx:idx+batch_size], target[idx:idx+batch_size]
            data_batch, target_batch = data_batch.to(device), target_batch.to(device)

            optimizer.zero_grad()

            output = self.forward(data_batch)
            loss = F.nll_loss(output, target_batch)
            loss.backward()
            optimizer.step()
            if verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_batch), len(target),
                    100. * batch_idx / int(len(data)/batch_size), loss.item()))
            batch_idx += 1

    def predict(self, data):
        """Predict
        ========

        Predict labels using trained neural network.
        
        Parameters
        ----------
        data : numpy array (float)
            Features

        Returns
        -------
        labels : numpy array (int)
            Predicted labels.
        """

        data = torch.from_numpy(data).float()
        self.to(torch.device("cpu"))
        self.eval()
        with torch.no_grad():
            pred = torch.argmax(self.forward(data), dim=1)
        labels = pred.numpy().astype(int)
        return labels

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



