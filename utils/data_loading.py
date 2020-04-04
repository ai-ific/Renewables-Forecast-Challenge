#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:16:11 2020

Functions for loading the data.

@author: José Enrique García, Verónica Sanz, Roberto Bruschini, 
         Carlos García, Salvador Tortajada, Pablo Villanueva (IFIC)
"""
import os
import pandas as pd

def load_dataframe(filename, path = './', **kwargs):
    '''
    Load a file into a dataframe
    '''
    if 'csv' in os.path.splitext(filename)[1]:
        df = pd.read_csv(path+filename, **kwargs)
    if 'xls' in os.path.splitext(filename)[1]:
        df = pd.read_excel(path+filename, **kwargs)
    return df

def clean(x):
    '''
    Parse dataframe columns to replace unwanted characters and leave just numbers
    '''
    try:
        return  x.str.replace(r"[a-zA-Z\%\/²]",'')
    except:
        return x
    
def wind_to_num(x):
    '''
    Parse the wind direction into a number
    0 = NaN, 
    1 = North, 2 = NNE, 3 = NE, 4 = ENE,
    5 = East, 6 = ESE, 7 = SE, 8 = SSE,
    9 = South, 10 = SSW, 11 = SW, 12 = WSW,
    13 = West, 14 = WNW, 15 = NW, 16 = NNW.
    '''
    dict = {'nan':0, 
            'North':1, 'NNE':2, 'NE':3, 'ENE':4, 
            'East':5, 'ESE':6, 'SE':7, 'SSE':8, 
            'South':9,'SSW':10, 'SW':11, 'WSW':12, 
            'West':13, 'WNW':14, 'NW':15, 'NNW':16}
    
    return list(map(dict.get, list(x)))
