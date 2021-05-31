from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def train_test_split():
    """Get only 13 cm range data

    Returns
    -------

    """
    data = pd.read_csv('labels.csv', sep=';')
    data_13 = data[data['Range'] == 13]

    x = np.array(data_13['Id']).reshape(-1, 1)
    y = np.array(data_13['B-lines'])
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for train_index, test_index in split.split(x, y):
        train_set = data_13.iloc[train_index]
        test_set = data_13.iloc[test_index]
    
    plt.hist([y, train_set['B-lines'], test_set['B-lines']], bins=[0, 1, 2], rwidth=0.5, label=['Whole set', 'Train set', 'Test set'], color=['tab:green','tab:red','tab:purple'])
    plt.xticks(ticks=[0.5, 1.5], labels=['No B-lines', 'B-lines'])
    plt.legend()
    plt.show()
    plt.savefig('figures/bline_ratio.png')
  
    return train_set, test_set
