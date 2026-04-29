import numpy as np
from ucimlrepo import fetch_ucirepo

def load_ucirepo_dataset(dataset_name):
    # fetch dataset
    energy_efficiency = fetch_ucirepo(id=242)

    # data (as pandas dataframes)
    X = energy_efficiency.data.features
    y = energy_efficiency.data.targets

    '''    # metadata
    print(energy_efficiency.metadata)

    # variable information
    print(energy_efficiency.variables)'''

    return X, y
  



