#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:19:55 2020

Data visualization

@author: José Enrique García, Verónica Sanz, Roberto Bruschini, 
         Carlos García, Salvador Tortajada, Pablo Villanueva (IFIC)
"""

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, read_csv

def plot_one_var(df, var):
    '''
    Plot one variable of a dataframe
    '''
    time_val = df['DateRound'].values
    variable = df[var].values
    
    plt.figure(figsize = (6,4))
    plt.plot(time_val, variable)
    plt.title(var)
    plt.xlabel('Time')
    plt.ylabel(var)
    plt.show()
    
def autocorrelation(x,y):
    corr = np.correlate(x, y, 'valid')
    #norm = np.correlate(x, x, 'valid')
    corr = corr[corr.size//2:]
    #corr = corr/np.max(norm)
    return corr

def fit_history_plot(history):
    '''
    Display the training (fit) history of a model

    '''
    # Display the model performance
    if 'mse' in history.history.keys():
        fig = plt.figure(figsize=(20,10), dpi= 80)
        fig.add_subplot(1,2,1)
        # minimum value and argument
        argmin_ = np.argmin(history.history['val_loss'])
        min_ = np.min(history.history['val_loss'])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.scatter(argmin_, min_, s=80, facecolors='none', edgecolors='r')
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        string_ = '{arg}, {min:.2}'.format(arg=argmin_+1, min=min_)
        plt.text(x = (argmin_*1.02), y = (min_*.98), s=string_, color='r')
        plt.grid()
        
        fig.add_subplot(1,2,2)
        plt.plot(history.history['mse'])
        plt.plot(history.history['val_mse'])
        min_ = history.history['val_mse'][argmin_]
        plt.scatter(argmin_, min_, s=80, facecolors='none', edgecolors='r')
        plt.title('Model MSE')
        plt.ylabel('mse')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        string_ = '{arg}, {min:.2}'.format(arg=argmin_+1, min=min_)
        plt.text(x = (argmin_+1), y = (min_*1.01), s=string_, color='r')
        plt.grid()
        
    else:
        fig = plt.figure(figsize=(10,5), dpi= 80)
        argmin_ = np.argmin(history.history['val_loss'])
        min_ = np.min(history.history['val_loss'])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.scatter(argmin_, min_, s=80, facecolors='none', edgecolors='r')
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        string_ = '{arg}, {min:.2}'.format(arg=argmin_+1, min=min_)
        plt.text(x = (argmin_*1.02), y = (min_*.98), s=string_, color='r')
        plt.grid()
        
    plt.show()
    
def money_plot(actual, predicted, xlab='Time', ylab='Demand'):
    '''
    Display a money plot to compare prediction value with actual values
    '''
    fig = plt.figure(figsize=(10,6), dpi=80)
    plt.plot(actual)
    plt.plot(predicted)
    plt.title('Money plot')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(['Actual '+ylab, 'Predicted '+ylab], loc='upper right')
    plt.grid()
    plt.show()

def boxplot_results(filenames):
    '''
    Display boxplot results from different test result files.
    '''
    results = DataFrame()
    for i in range(len(filenames)):
        if i==0:
            results = read_csv(filenames[i])
            results.columns = [filenames[i].replace('.csv','')]
        else:
            auxdf = read_csv(filenames[i])
            results[filenames[i].replace('.csv','')] = auxdf
    plt.figure(figsize=(12,8))
    results.boxplot()
    plt.title('Model results boxplots')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45, fontsize='small')
    plt.tight_layout()
    plt.show()
