# MachineLearningAnthro

## Loading data for machine learning

The `utils.py` file contains functions for loading the machine learning datasets as numerical datasets that can be directly used in machine learning. To load the break-level data, run
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

In both cases, there is an optional argument `standard_scaler`, which if set to true will center and scale all features to have zero mean and unit variance. To use the feature, pass `standard_scaler=True` into either dataset constructor, as below
```
from utils import frag_level_ml_dataset, break_level_ml_dataset

data,target,specimens,target_names = frag_level_ml_dataset(standard_scaler=True)
data,target,specimens,break_numbers,target_names = break_level_ml_dataset(standard_scaler=True)
```


## Other scripts
We need to describe all other scripts:
1. `moclan.py`:
2. `moclan_modified.py`:
3. `frag_level_experiment.py`: This is an experiment that compares train/test splits at fragment level versus break level.
