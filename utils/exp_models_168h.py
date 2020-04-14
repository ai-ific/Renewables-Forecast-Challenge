#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:25:44 2020

@author: 
"""
import warnings
warnings.filterwarnings("ignore")

import data_loading
import data_viz
import processing_utils
import deep_learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping #, ModelCheckpoint

if __name__ == "__main__":
    # loading data
    meteo_df = data_loading.load_dataframe('data/Meteo.csv', parse_dates=[['Date', 'Time']])
    xls_df = data_loading.load_dataframe('data/Datos_solar_y_demanda_residencial.xlsx')
    
    # common time
    meteo_df['DateRound'] = meteo_df['Date_Time'].dt.round('5min')
    xls_df['DateRound'] = xls_df['Date'].dt.round('5min')
    meteo_df['Hour'] = meteo_df['DateRound'].dt.hour
    meteo_df['Day'] = meteo_df['DateRound'].dt.dayofyear
    
    # Parse data
    meteo_df['WindNum'] = data_loading.wind_to_num(meteo_df['Wind'])
    meteo_df = meteo_df.apply(lambda x : data_loading.clean(x))
    
    # Drop columns
    meteo_df = meteo_df.drop(columns=['UV', 'Solar','Wind'])
    
    # Filter variables
    meteo_filtered = meteo_df[['DateRound', 'Temperature', 'Dew Point','Humidity', 'Speed', 'Gust', 'Pressure', 'Precip. Rate.', 'Precip. Accum.','WindNum','Hour', 'Day']]
    xls_filtered = xls_df[['DateRound', 'Demanda (W)']]
    
    # Merge dataframes
    final_df = pd.merge(xls_filtered, meteo_filtered, on='DateRound')
    final_df = final_df.drop(columns=['DateRound'])
    
    # A correlation matrix to understand the data
    final_df = final_df.astype(np.float32)
    correlation = final_df.corr()
    
    # Deep learning model setting up, callback definition and training
    # model set up
    n_layers = 1
    n_units = 50
    epochs= 50
    batch_size = 64
    
    # callback
    # best_model_filename = 'best_LSTM_model_{epoch:02d}.h5'
    #checkpointer = ModelCheckpoint(best_model_filename, monitor='val_loss', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    EPOCHS = 50
    BATCHSIZE = 64
    result_filenames = list()
    '''
    7th Experiment. Model with 6 features, 168 hours
    Input: Features using 168 previous hours to hour t
    Output: Energy demand of hour t
    Network: One LSTM layer with 12 units
    '''
    time_steps = 168
    n_units = 12
    selected_columns = ['Hour', 'Temperature', 'Dew Point', 'Humidity', 'Pressure', 'Demanda (W)']
    n_features = len(selected_columns)
    working_df = final_df[selected_columns]
    norm_dataset, scaler = processing_utils.normalize_data(working_df)
    dataset = processing_utils.series_to_supervised(norm_dataset, n_in=time_steps, n_out=1)
    data_training, data_test = train_test_split(dataset, train_size=.9, shuffle=True, random_state=20132018)
    Xtr, Ytr = processing_utils.dataset_to_XY(data_training, n_features)
    Xte, Yte = processing_utils.dataset_to_XY(data_test, n_features)
    Xtr = Xtr.reshape(Xtr.shape[0], time_steps, n_features)
    Xte = Xte.reshape(Xte.shape[0], time_steps, n_features)    
    model = deep_learning.LSTMNet(units=n_units, 
                                  input_size=(Xtr.shape[1], Xtr.shape[2]))
    rmse_df = deep_learning.run_experiment(model, Xtr, Ytr, Xte, Yte,
                   callback = [earlystopper],
                   epochs = EPOCHS,
                   batch_size = BATCHSIZE,
                   repeats = 30,
                   validation_split = .1,
                   scaler = scaler, 
                   verbose = True)
    result_filenames.append('model_6f_168h_12_shuffle.csv')
    rmse_df.to_csv(result_filenames[-1], index=False)
    
    '''
    8th Experiment. Model with 6 features, 168 hours II
    Input: Features using 168 previous hours to hour t
    Output: Energy demand of hour t
    Network: One LSTM layer with 50 units
    '''
    time_steps = 168
    n_units = 50
    selected_columns = ['Hour', 'Temperature', 'Dew Point', 'Humidity', 'Pressure', 'Demanda (W)']
    n_features = len(selected_columns)
    working_df = final_df[selected_columns]
    norm_dataset, scaler = processing_utils.normalize_data(working_df)
    dataset = processing_utils.series_to_supervised(norm_dataset, n_in=time_steps, n_out=1)
    data_training, data_test = train_test_split(dataset, train_size=.9, shuffle=True, random_state=20132018)
    Xtr, Ytr = processing_utils.dataset_to_XY(data_training, n_features)
    Xte, Yte = processing_utils.dataset_to_XY(data_test, n_features)
    Xtr = Xtr.reshape(Xtr.shape[0], time_steps, n_features)
    Xte = Xte.reshape(Xte.shape[0], time_steps, n_features)    
    model = deep_learning.LSTMNet(units=n_units, 
                                  input_size=(Xtr.shape[1], Xtr.shape[2]))
    rmse_df = deep_learning.run_experiment(model, Xtr, Ytr, Xte, Yte,
                   callback = [earlystopper],
                   epochs = EPOCHS,
                   batch_size = BATCHSIZE,
                   repeats = 30,
                   validation_split = .1,
                   scaler = scaler, 
                   verbose = True)
    result_filenames.append('model_6f_168h_50_shuffle.csv')
    rmse_df.to_csv(result_filenames[-1], index=False)
    
    '''
    9th Experiment. Model with 6 features III
    Input: Features using 168 previous hours to hour t
    Output: Energy demand of hour t
    Network: LSTM layers with 50 and 10 units
    '''
    time_steps = 168
    n_units = [50, 10]
    selected_columns = ['Hour', 'Temperature', 'Dew Point', 'Humidity', 'Pressure', 'Demanda (W)']
    n_features = len(selected_columns)
    working_df = final_df[selected_columns]
    norm_dataset, scaler = processing_utils.normalize_data(working_df)
    dataset = processing_utils.series_to_supervised(norm_dataset, n_in=time_steps, n_out=1)
    data_training, data_test = train_test_split(dataset, train_size=.9, shuffle=True, random_state=20132018)
    Xtr, Ytr = processing_utils.dataset_to_XY(data_training, n_features)
    Xte, Yte = processing_utils.dataset_to_XY(data_test, n_features)
    Xtr = Xtr.reshape(Xtr.shape[0], time_steps, n_features)
    Xte = Xte.reshape(Xte.shape[0], time_steps, n_features)    
    model = deep_learning.LSTMNet(units=n_units, 
                                  input_size=(Xtr.shape[1], Xtr.shape[2]))
    model.summary()
    rmse_df = deep_learning.run_experiment(model, Xtr, Ytr, Xte, Yte,
                   callback = [earlystopper],
                   epochs = EPOCHS,
                   batch_size = BATCHSIZE,
                   repeats = 30,
                   validation_split = .1,
                   scaler = scaler, 
                   verbose = True)
    result_filenames.append('model_6f_168h_50_10_shuffle.csv')
    rmse_df.to_csv(result_filenames[-1], index=False)
    

    data_viz.boxplot_results(result_filenames)

    '''
    result_filenames = ['baseline_12.csv', 'baseline_50.csv', 'baseline_50_10.csv',
                        'model_6f_12.csv', 'model_6f_50.csv', 'model_6f_50_10.csv', 
                        'model_6f_168h_12.csv', 'model_6f_168h_50.csv', 'model_6f_168h_50_10.csv']
    
    data_viz.boxplot_results(result_filenames)
    '''
    
    
