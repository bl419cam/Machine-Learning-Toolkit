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

def plot_confusion_matrix(y_true, y_pred, model_name='', cmap=plt.cm.Blues):
    """
    plot confusion matrix for array of true labels and array of predictions from 
    a classifier model
    """
    #initialize confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_norm = metrics.confusion_matrix(y_true, y_pred, normalize='true')

    #turn off gridlines (if any)
    plt.grid(b=None)
    #plot basic matrix
    plt.imshow(cm_norm, cmap=cmap)

    #add title and axis labels
    plt.title('Confusion Matrix {}'.format(model_name))
    plt.xlabel('Predictions')
    plt.ylabel('True Labels')

    #add axis scale and markers
    class_names = set(np.unique(y_true))
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)

    #format matrix
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j], horizontalalignment='center', fontdict={'size':12},
                    color='white' if cm_norm[i,j] > 0.5 else 'black')

def plot_train_test_roc_curve(y_test, y_test_score, y_train, y_train_score, 
                    clf_name='Binary Classifier'):
    """
    plot roc curve for training and test sets on the same graph
    """
    plt.style.use('ggplot')
    colors = sns.color_palette('Set2')

    fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_test_score)
    fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, y_train_score)

    plot.figure(figsize=(8,6))
    plt.plot([0,1], [0,1], linestyle='--', label=='random')
    plt.plot(fpr_train, tpr_train, color=colors[1], marker='.', label='train set')
    plt.plot(fpr_test, tpr_test, color=colors[0], marker='.', label='test set')
    plt.title('ROC Curve - {}'.format(clf_name))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend(loc='lower right')

    return plt

def plot_train_test_precision_recall_curve(y_test, y_test_score, y_train, y_train_score, 
                    clf_name='Binary Classifier'):
    """
    plot precision-recall curve for training and test sets on the same graph
    """
    plt.style.use('ggplot')
    colors = sns.color_palette('Set2')

    pr_test, rc_test, thresholds_test = metrics.precision_recall_curve(y_test, y_test_score)
    pr_train, rc_train, thresholds_train = metrics.precision_recall_curve(y_train, y_train_score)

    imb_test = sum(y_test==1)/len(y_test)
    imb_train = sum(y_train==1)/len(y_train)
    imb_avg = (imb_test + imb_train)/2

    plt.figure(figsize(8,6))
    plt.plot([0,1], [imb_avg, imb_avg], linestyle='--', label='random')
    plt.plot(rc_train, pr_train, color=colors[1], marker='.', label='train set')
    plt.plot(rc_test, pr_test, color=colors[0], marker='.', label='test set')
    plt.title('Precision-Recall Curve - {}'.format(clf_name))
    plt.xlabel('Recall (True Positive Rate')
    plt.ylabel('Precision')
    plt.legend()

    return plt

def find_threshold_by_recall(y_labels, y_score, recall):
    
    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_score)
    ix = np.where(np.logical_and(tpr>=recall, tpr<(recall+0.1)))[0][0]

    return thresholds[ix]

def find_kstat(y_labels, y_score):

    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_score)
    kstat = max(tpr-fpr)
    kstat_thresh = threshoolds[np.argmax(tpr-fpr)]

    return kstat, kstat_thresh