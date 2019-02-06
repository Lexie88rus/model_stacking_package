#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 23:32:56 2019

@author: aleksandra deis

Module contains helper methods for model stacking implemrntation.
Stacking implementation is based on Kaggle's kernel:
https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

"""

#imports

import numpy as np #import linear algebra
from sklearn.model_selection import KFold #import folding from sklearn

#define constants for folding

""" Number of folds """
NFOLDS = 5

""" Random seed for folding """
SEED = 42


"""

Class to wrap the Sklearn classifier in order to pass different
classifiers to folding function
 
"""
class SklearnHelper(object):
    """ 
    Init function
    
    INPUT
        clf - sklearn classifier
        seed - random seed
        params - dictionary containing parameters for the classifier clf
    """
    def __init__(self, clf, seed=SEED, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    """
    Wrapper for train function
    
    INPUT
        x_train - training set
        y_train - response vector for the training set
        
    RETURNS
        trained model
    """
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    """
    Wrapper for predict function
    
    INPUT
        x - training set
        
    RETURNS
        predicted response vector for x
    """
    def predict(self, x):
        return self.clf.predict(x)
    
    """
    Wrapper for feature impornaces function
    
    INPUT
        x - training set
        y  - response vector for the training set
        
    RETURNS
        feature importances for trained model using x and y
    """
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)        

"""
Function, which performs out of fold train and test predictions, 
which will be used as new variables for stacking models.

INPUT
    clf - warpped sklearn classifier
    x_train - training dataset
    y_train - response vector for training dataset
    x_test - test dataset
    n_folds - number of folds for stacking
    seed - random seed

"""
def get_oof(clf, x_train, y_train, x_test, nfolds = NFOLDS, seed = SEED):
    arg = {'n_splits':nfolds, 'random_state':seed}
    kf = KFold( **arg)
    
    
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((nfolds, x_test.shape[0]))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
