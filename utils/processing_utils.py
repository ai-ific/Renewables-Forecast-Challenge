#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:50:39 2020

Utils for preparing data

@author: José Enrique García, Verónica Sanz, Roberto Bruschini, 
         Carlos García, Salvador Tortajada, Pablo Villanueva (IFIC)
"""
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, concat
import numpy as np

def col_index(df, name):
    '''
    Index number of column by name
    '''
    return df.columns.to_list().index(name)

def normalize_data(df):
    '''
    Returns data normalized using minmax 0-1 normalization
    '''
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    return normalized_data, scaler

def series_to_supervised_with_actual_hour(data, n_in=1, n_out=1, dropnan=True):
    '''
    Prepare series to supervised learning using the hour of the prediction demand
    '''
    cols, names = list(), list()
    
    # Dataframe without Hour...
    df_to_shift = DataFrame(data[:,1:])
    n_vars = df_to_shift.shape[1]
    
    # Shift other vars to t-1 and append
    for i in range(n_in, 0, -1):
        cols.append(df_to_shift.shift(i))
        names += [('var{}(t-{})'.format(j+1, i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df_to_shift.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # add Hours
    df_hour = DataFrame(data[:,0])
    cols.insert(0, df_hour[0])
    #cols.append(df_hour[0])
    names = [('Hours(t)')] + names
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    '''
    Prepare series to supervised learning
    '''
    cols, names = list(), list()
    
    # Dataframe without Hour...
    df = DataFrame(data)
    n_vars = df.shape[1]
    
    # Shift other vars to t-1 and append
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var{}(t-{})'.format(j+1, i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def dataset_to_XY(dataset, nvars, target_index = -1):
    '''
    Splits a dataset into its input feature array X and its target array Y

    Parameters
    ----------
    dataset : DataFrame, list or ndarray.
    nvars : int. Number of features to include as input X.
    target_index : int, optional. Location of the target variable. The default is -1.

    Returns
    -------
    X : ndarray. Input array.
    Y : ndarray. Target array.
    '''
    index = nvars
    if isinstance(dataset, DataFrame):
        dataset_ = dataset.values
        X, Y = dataset_[:,:-index], dataset_[:, target_index]
    if isinstance(dataset, (list, np.ndarray)):
        X, Y = dataset[:,:-index], dataset[:, target_index]
    return X, Y

def target_inverse_scale(Y, scaler, index = -1):
    '''
    Unnormalizes the normalized target output of the neural network to get the 
    unnormalized/original target values

    Parameters
    ----------
    Y : array of target (actual or predicted) values.
    scaler : a scikit learn MinMaxScaler.
    index : int, optional. The index position in the scaler of the feature to 
    be unnormalized. The default is -1.

    Returns
    -------
    unnormalized_Y : an array with the unnormalized target values.

    '''
    single_inverse_scale = MinMaxScaler()
    single_inverse_scale.min_, single_inverse_scale.scale_ = scaler.min_[index], scaler.scale_[index]
    unnormalized_Y = single_inverse_scale.inverse_transform(Y)
    return unnormalized_Y

def prediction_correction(y, yhat):
    '''
    Do not use.
    '''
    correction = np.mean(y/yhat)
    return yhat*correction
