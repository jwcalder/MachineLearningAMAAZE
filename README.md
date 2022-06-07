# MachineLearningAMAAZE

This repository provides the code and data to reproduce the experimental results from the paper

K. Yezzi-Woodley, A. Terwilliger, J. Li, E. Chen, M. Tappen, J. Calder, P. J. Olver. [Using machine learning on new feature sets extracted from 3D models of broken animal bones to classify fragments according to break agent.](https://arxiv.org/abs/2205.10430) arXiv preprint:2205.10430, 2022.

##General overview

The folder `preprocessing` contains all the python scripts used to preprocess the raw data we collected into a form that can be used by machine learning algorithms. The scripts in the preprocessing folder generate the files `break_level_ml.csv` and `frag_level_ml.csv`, which contain the break-level and fragment-level datasets, respectively, with all data converted to numerical formats. 

The file `moclan.csv` contains the data from Moclan et al., 2019, which we compare against, and `moclan_carngrouped.csv` contains the same data except with the carnivores grouped together.

All results are saved in .csv files in the `results` folder. Likewise, many of the tables from the paper are automatically generated and saved in the `tables` folder, and figures are saved in the `figures` folder. 

The python scripts require the installation of a number of python packages, all listed in the `requirements.txt` file. To install all required packages run
```
pip install -r requirements.txt
```

## Description of scripts

1. `ml_test.py` runs all the main machine learning tests in the paper.
2. `Moclan_replication.py` runs all of the experiments from our paper concerned with replicating the Moclan et al., 2019 study.
3. `SpectralEmbedding.py` runs the unsupervised graph-based learning experiments, including spectral embeddings and clusterings.
4. `randomized_experiment.py` runs the experiment from our paper with applying bootstrapping and break-level train-test splits on randomized datasets.
5. `randomized_experiment_table.py` generates a table summarizing the results from the randomized experiment.
6. `utils.py` contains a variety of utility functions that are used over multiple scripts.

## Loading data for machine learning

The `utils.py` file contains functions for loading our datasets as numerical datasets that can be directly used in machine learning. To load the break-level data, run
```
from utils import break_level_ml_dataset

data,target,specimens,break_numbers,target_names = break_level_ml_dataset()
```
The first two outputs `data` and `target` are the features and labels for the dataset, all converted to numerical values. The final outputs `specimens`, `break_numbers`, and `target_names` are lists of specimen names and break numbers for all datapoints (breaks) and the target names for each numerical label. The function has several optional arguments to configure how the dataset is built, and the field used for targets; see the documentation in `utils.py` for more information.

To load the fragment-level dataset, run the code
```
from utils import frag_level_ml_dataset

data,target,specimens,target_names = frag_level_ml_dataset()
```


## Contact and questions

Email <jwcalder@umn.edu> with any questions or comments.

