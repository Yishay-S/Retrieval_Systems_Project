from config import *
import pandas as pd
from sklearn import preprocessing
def get_data():
    """
    reads train, validation to python indices so we don't need to deal with it in each algorithm.
    of course, we 'learn' the indices (a mapping from the old indices to the new ones) only on the train set.
    if in the validation set there is an index that does not appear in the train set then we can put np.nan or
     other indicator that tells us that.
    """
    train = pd.read_csv(TRAIN_PATH)
    train['train_val'] = 'train'
    validation = pd.read_csv(VALIDATION_PATH)
    validation['train_val'] = 'val'
    train_val = pd.concat([train, validation]).reset_index()
    le = preprocessing.LabelEncoder()
    train_val[USER_COL_NAME_IN_DATAEST] =le.fit_transform(train_val[USER_COL_NAME_IN_DATAEST])
    train_val[ITEM_COL_NAME_IN_DATASET] =le.fit_transform(train_val[ITEM_COL_NAME_IN_DATASET])
    train = train_val[train_val['train_val'] == 'train'].drop(columns=['train_val']).reset_index(drop=True)#
    validation = train_val[train_val['train_val'] == 'val'].drop(columns=['train_val']).reset_index(drop=True)#
    return train, validation

class Config:
    def __init__(self, **kwargs):
        self._set_attributes(kwargs)

    def _set_attributes(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)