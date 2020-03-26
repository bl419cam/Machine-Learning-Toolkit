"""
custom functions for machine learning projects
"""
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

def onehotencode(X, keep_index=False):
    """
    One hot encode categorical variables for a single dataframe
    """
    X_ind = X.index
    X_obj = X[[col for col, dtype in list(zip(X.columns, X.dtypes)) if dtype == np.dtype('O')]]
    X_nonobj = X[[col for col, dtype in list(zip(X.columns, X.dtypes)) if dtype != np.dtype('O')]]

    ohe = OneHotEncoder(handle_unknown='ignore')
    X_obj_ohe = ohe.fit_transform(X_obj)

    X_nonobj_df = pd.DataFrame(X_nonobj).reset_index(drop=True)
    X_obj_ohe_df = pd.DataFrame(X_obj_ohe.todense(), columns=ohe.get_feature_names()).reset_index(drop=True)

    X_all = pd.concat([X_nonobj_df, X_obj_ohe_df], axis=1)

    if keep_index:
        X_all.index = X_ind

    return X_all

def onehotencode_train_test(train, test, keep_index=False):
    """
    Take train and test datasets in the form of panda dataframes, 
    onehotencode categorical in both train and test sets by 
    fitting to train set ony. Return both datasets with onehotencoded variables.
    If keep_index True, original index in train and test will be kept, otherwise
    index will be reset for both.
    """
    train_ind = train.index
    test_ind = test.index

    train_obj = train[[col for col, dtype in list(zip(train.columns, train.dtypes))
                                                if dtype == np.dytpe('O')]]
    train_nonobj = train[[col for col, dtype in list(zip(train.columns, train.dtypes))
                                                if dtype != np.dytpe('O')]]
    
    test_obj = test[[col for col, dtype in list(zip(test.columns, test.dtypes))
                                                if dtype == np.dytpe('O')]]
    test_nonobj = test[[col for col, dtype in list(zip(test.columns, test.dtypes))
                                                if dtype != np.dytpe('O')]]

    ohe = OneHotEncoder(hande_unknown='ignore')
    train_obj_ohe = ohe.fit_transform(train_obj)

    train_nonobj_df = pd.DataFrame(train_nonobj).reset_index(drop=True)
    train_obj_ohe_df = pd.DataFrame(train_obj_ohe.todense(), columns=ohe.get_feature_names()).reset_index(drop=True)
    train_all = pd.concat([train_nonobj_df, train_obj_ohe_df], axis=1)

    test_obj_ohe = ohe.transform(test_obj)
    test_nonobj_df = pd.DataFrame(test_nonobj).reset_index(drop=True)
    test_obj_ohe_df = pd.DataFrame(test_obj_ohe.todense(), columns=ohe.get_feature_names()).reset_index(drop=True)
    test_all = pd.concat([test_nonobj_df, test_obj_ohe_df], axis=1)

    if keep_index:
        train_all.index = train_ind
        test_all.index = test_ind
    
    return train_all, test_all