#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 19:56:25 2020

Functions and utilities for LSTM recurrent networks

@author: José Enrique García, Verónica Sanz, Roberto Bruschini, 
         Carlos García, Salvador Tortajada, Pablo Villanueva (IFIC)
"""
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.initializers import glorot_uniform
import keras.backend as K
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import DataFrame
import processing_utils

def LSTMNet(units, input_size, use_dropout=False):
    '''
    Build an LSTM neural network model

    Parameters
    ----------
    units : int, tuple or list. The number of units of each LSTM layer.
    input_size : list. The input shape of the input tensor.
    use_dropout : boolean, optional. Whether to use or not a dropout layer 
    after the last LSTM layer. The default is False.
    TO DO : select different optimizers (use **kwargs)

    Returns
    -------
    model : An LSTM neural network model designed and compiled

    '''
    model = Sequential()
    
    if isinstance(units, int):
        model.add(LSTM(units, input_shape = input_size, return_sequences=False))
            
    if isinstance(units, (tuple, list)):
        for i in range(len(units)):
            if i==0:
                model.add(LSTM(units[i], input_shape = input_size, return_sequences=True))
            else:
                if i!=len(units)-1:
                    model.add(LSTM(units[i], return_sequences=True))
                else:
                    model.add(LSTM(units[i], return_sequences=False))
    
    if use_dropout:
        model.add(Dropout(0.5))
        
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss = 'mae', optimizer = 'Adam', metrics = ['mse'])
    
    return model

def restart_weights(model):
    '''
    Restart the weights of a model using Glorot uniform weight initialization. 
    Returns the list of weights.

    '''
    old_w = model.get_weights()
    k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
    new_w = [k_eval(glorot_uniform()(w.shape)) for w in old_w]
    return new_w

def run_experiment(model, Xtr, Ytr, Xte, Yte,
                   callback,
                   epochs,
                   batch_size,
                   repeats,
                   validation_split,
                   scaler, 
                   verbose = False):
    '''
    Run an experiment: fit a model with training data, using validation split, 
    repeating n times the split, and assessing its performance using root mean
    square error with an independent test set. This returns a dataframe with 
    n RMSErrors.
    '''
    
    Yact = processing_utils.target_inverse_scale(Yte.reshape(-1,1), scaler)
    
    rmse_list = list()
    
    for i in range(repeats):
        # training
        history = model.fit(Xtr, Ytr, epochs=epochs, batch_size=batch_size, validation_split=validation_split, 
                        verbose=0, shuffle=False, callbacks = callback)
    
        if model.stateful:
            model.reset_states()
        
        # Evaluate the model with the independent test set
        yhat_norm = model.predict(Xte)
        
        # restart model weights
        model.set_weights(restart_weights(model))
    
        # Scaler inverse transform
        Yhat = processing_utils.target_inverse_scale(yhat_norm, scaler)
    
        # Compute RMSE and add to the list of root mean squared errors
        rmse = sqrt(mean_squared_error(Yact, Yhat))
        rmse_list.append(rmse)
        
        # verbose information
        if verbose:
            print('Repeat {} - RMSE = {}'.format(i+1, rmse))
        
    return DataFrame(rmse_list)

