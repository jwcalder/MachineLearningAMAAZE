import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import amaazetools.trimesh as tm
from sklearn import preprocessing
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plt
import os,sys
import pandas as pd

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
frag_level_fields_categorical = ['trab']

#Frag level numerical fields
frag_level_fields_numerical = ['Surface Area',
                               'Volume',
                               'Bounding Box Dim1',
                               'Bounding Box Dim2',
                               'Bounding Box Dim3']


def break_level_ml_dataset(numerical_fields=None, categorical_fields=None, target_field='Effector'):
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

    df = pd.read_csv('../data/break_level_ml.csv') 

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

    return data,target,specimens,break_numbers,target_names

def frag_level_ml_dataset(numerical_fields=None, categorical_fields=None, sum_stats_fields=None,
                          sum_stats=None, count_fields=None, target_field='Effector'):
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

    df = pd.read_csv('../data/frag_level_ml.csv') 

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

    return data,target,specimens,target_names


        

class Net(nn.Module):
    def __init__(self, structure=[10,10], num_classes=2, dropout_rate=0.0, batch_normalization=False,
                 epochs=100, cuda=False, learning_rate=0.1,batch_size=32, gamma=0.9, verbose=False):
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
        dropout_rate : float (optional), default=0
            Dropout rate.
        batch_normalization : bool (optional), default=True
            Whether to apply batch normalization.
        epochs : int (optional), default=100
            Number of training epochs (loops over whole dataset).
        cuda : bool (optional), default=False
            Whether to use GPU, if found.
        learning_rate : float (optional), default = 0.1
            Learning rate for training.
        batch_size : int (optional), default = 32
            Size of mini-batches for stochastic gradient descent.
        gamma : float (optional), default = 0.9
            Scheduler parameter. How much to decrease learning rate at each iteration.
        verbose : bool (optional), default = False
            Whether to print out details during training or not.
        """

        super(Net, self).__init__()

        self.structure = structure
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.epochs = epochs
        self.cuda = cuda
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.verbose = verbose

        self.num_layers = len(structure)-1
        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(self.num_layers):
            self.fc.append(nn.Linear(structure[i],structure[i+1]))
            self.bn.append(nn.BatchNorm1d(structure[i]))
        self.final = nn.Linear(structure[self.num_layers],num_classes)

        self.dropout = nn.Dropout(dropout_rate)

    def get_params(self, deep=True):
        return {'structure':self.structure,
                'num_classes':self.num_classes,
                'dropout_rate':self.dropout_rate,
                'batch_normalization':self.batch_normalization,
                'epochs':self.epochs,
                'cuda':self.cuda,
                'learning_rate':self.learning_rate,
                'batch_size':self.batch_size,
                'gamma':self.gamma,
                'verbose':self.verbose}

    def __str__(self):
        s =  'NeuralNetwork('
        s += 'structure='+str(self.structure)
        s += ',num_classes='+str(self.num_classes)
        s += ',dropout_rate='+str(self.dropout_rate)
        s += ',batch_normalization='+str(self.batch_normalization)
        s += ',epochs=%d'%self.epochs
        s += ',cuda='+str(self.cuda)
        s += ',learning_rate='+str(self.learning_rate)
        s += ',batch_size=%d'%self.batch_size
        s += ',gamma='+str(self.gamma)
        s += ',verbose='+str(self.verbose)
        s += ')'
        return s

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

    def fit(self, data, target):
        """Fit
        ========

        Trains the neural network.
        
        Parameters
        ----------
        data : numpy array (float)
            Features
        target : numpy array (int)
            Labels.
        """

        #Reset weights
        self.reset()

        #Convert to torch
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).long()

        #Cuda (GPU)
        use_cuda = self.cuda and torch.cuda.is_available()
        if self.verbose:
            if use_cuda:
                print("Using GPU!")
            else:
                print("Using CPU.")
        device = torch.device("cuda" if use_cuda else "cpu")
        self.to(device)

        #Optimizer and scheduler
        optimizer = optim.Adadelta(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)

        for epoch in range(1, self.epochs + 1):
            if self.verbose:
                print('\nEpoch: %d'%epoch)
            self.train_epoch(device, data, target, optimizer, epoch, self.batch_size, self.verbose)
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

def train_test_split(data, target, specimens, percent_test = 0.25):
    """Train Test Split
    ========
    
    Converts the data and target arrays into training and testing datasets.
    Splits on specimen with the desired percentage of them in the testing dataset. 
    This will fail if the inputs aren't in the same order. 
    
    Parameters
    ----------
    data : numpy array (float)
        Features.
    target : numpy array (int)
        Targets as integers
    specimens : numpy array (string)
        List of specimen names.
    percent_test : float (optional), default = 0.1
        The percentage of the dataset that goes into the training dataset.
    
    Returns
    -------
    data_train : numpy array (float)
        Features in the train dataset.
    target_train : numpy array (int)
        Train targets as integers.
    data_test : numpy array (float)
        Features in the test dataset.
    target_test : numpy array (int)
        Test Targets as integers.
    frag_test : numpy array (string)
        Fragment names used for final voting.
    """
    
    # Create our list of frags
    fragments = np.unique(specimens)
    # Choose which ones go into our test set
    test_size = max(1, round(fragments.size * percent_test))
    testing_fragments = np.random.choice(fragments,size = test_size, replace=False)
    
    # Set up the training and testing datasets    
    data_train = []
    target_train = []
    data_test = []
    target_test = []
    frag_test = []
    
    # Split the dataset
    for i, frag in enumerate(specimens):
        if frag in testing_fragments:
            data_test.append(data[i])
            target_test.append(target[i])
            frag_test.append(frag)
        else:
            data_train.append(data[i])
            target_train.append(target[i])
    
    # Swap to numpy arrays
    data_train = np.array(data_train)
    target_train = np.array(target_train)
    data_test = np.array(data_test)
    target_test = np.array(target_test)
    frag_test = np.array(frag_test)
    
    # Randomization of the order of the data is just a good practice

    # data_train, target_train = shuffle(data_train, target_train)
    # data_test, target_test, frag_test = shuffle(data_test, target_test, frag_test)
    
    return data_train, target_train, data_test, target_test, frag_test

def specimen_voting(target_output, target_test, frag_test):
    """Train Test Split
    ========
    
    Compares outputted targets to their true labels, and votes within specimens.
    
    Parameters
    ----------
    target_output : numpy array (int)
        The outputted targets.
    target_test : numpy array (int)
        True targets.
    frag_test : numpy array (string)
        List of specimen names.
    
    Returns
    -------
    voting : dictionary
        Contains each fragment, its true label, and the guess from voting
    """
    
    # Populate the dictionary with truth values
    voting = {"frags": {}}
    for i, frag in enumerate(frag_test):
        voting["frags"][frag] = {"Truth" : target_test[i], "Votes" : [], "Guess" : None}
    
    # Add in the votes
    for i, output in enumerate(target_output):
        voting["frags"][frag_test[i]]["Votes"].append(output)
    
    # Total and sum accuracy
    correct = 0
    for frag in voting["frags"].keys():
        # This helps shake up the votes in case of ties.
        vote = np.argmax(np.bincount(voting["frags"][frag]["Votes"]) + 1e-6*np.random.rand(1,1))
        voting["frags"][frag]["Guess"] = vote
        if(voting["frags"][frag]["Guess"] == voting["frags"][frag]["Truth"]):
            correct += 1

    voting["Mean Accuracy"] = correct / len(voting["frags"].keys())
    
    return voting
