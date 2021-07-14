'''
DL models (FNN, 1D CNN and CNN-LSTM) evaluation on N-CMAPSS
12.07.2021
Hyunho Mo
hyunho.mo@unitn.it
'''
## Import libraries in python
import os
import json
import logging
import sys
import h5py
import time
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import random
import importlib
from scipy.stats import randint, expon, uniform
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
import scipy.stats as stats
# from sklearn.utils.testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning
# import keras
import tensorflow as tf
print(tf.__version__)
# import keras.backend as K
import tensorflow.keras.backend as K
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding
from tensorflow.keras.layers import BatchNormalization, Activation, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
seed = 0
random.seed(0)
np.random.seed(seed)
# Ignore tf err log
pd.options.mode.chained_assignment = None  # default='warn'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')




def main():
    # Time tracking, Operation time (min):  0.003
    t = time.process_time()

    # Load data
    '''
    W: operative conditions (Scenario descriptors)
    X_s: measured signals
    X_v: virtual sensors
    T(theta): engine health parameters
    Y: RUL [in cycles]
    A: auxiliary data
    '''

    with h5py.File(data_filepath, 'r') as hdf:
        # Development(training) set
        W_dev = np.array(hdf.get('W_dev'))  # W
        X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s
        X_v_dev = np.array(hdf.get('X_v_dev'))  # X_v
        T_dev = np.array(hdf.get('T_dev'))  # T
        Y_dev = np.array(hdf.get('Y_dev'))  # RUL
        A_dev = np.array(hdf.get('A_dev'))  # Auxiliary

        # Test set
        W_test = np.array(hdf.get('W_test'))  # W
        X_s_test = np.array(hdf.get('X_s_test'))  # X_s
        X_v_test = np.array(hdf.get('X_v_test'))  # X_v
        T_test = np.array(hdf.get('T_test'))  # T
        Y_test = np.array(hdf.get('Y_test'))  # RUL
        A_test = np.array(hdf.get('A_test'))  # Auxiliary

        # Varnams
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))
        X_v_var = np.array(hdf.get('X_v_var'))
        T_var = np.array(hdf.get('T_var'))
        A_var = np.array(hdf.get('A_var'))

        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype='U20'))
        X_s_var = list(np.array(X_s_var, dtype='U20'))
        X_v_var = list(np.array(X_v_var, dtype='U20'))
        T_var = list(np.array(T_var, dtype='U20'))
        A_var = list(np.array(A_var, dtype='U20'))


    W = np.concatenate((W_dev, W_test), axis=0)
    X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
    X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
    T = np.concatenate((T_dev, T_test), axis=0)
    Y = np.concatenate((Y_dev, Y_test), axis=0)
    A = np.concatenate((A_dev, A_test), axis=0)

    print('')
    print("Operation time (min): ", (time.process_time() - t) / 60)
    print("number of training samples(timestamps): ", Y_dev.shape[0])
    print("number of training samples(timestamps): ", Y_test.shape[0])
    print('')
    print("W shape: " + str(W.shape))
    print("X_s shape: " + str(X_s.shape))
    print("X_v shape: " + str(X_v.shape))
    print("T shape: " + str(T.shape))
    print("Y shape: " + str(Y.shape))
    print("A shape: " + str(A.shape))

    '''
    Illusration of Multivariate time-series of condition monitoring sensors readings for Unit5 (fifth engine)

    W: operative conditions (Scenario descriptors) - ['alt', 'Mach', 'TRA', 'T2']
    X_s: measured signals - ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
    X_v: virtual sensors - ['T40', 'P30', 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi']
    T(theta): engine health parameters - ['fan_eff_mod', 'fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod', 'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod', 'LPT_eff_mod', 'LPT_flow_mod']
    Y: RUL [in cycles]
    A: auxiliary data - ['unit', 'cycle', 'Fc', 'hs']
    '''
    df_W = DataFrame(data=W, columns=W_var)
    df_Xs = DataFrame(data=X_s, columns=X_s_var)
    df_Xv = DataFrame(data=X_v, columns=X_v_var)
    df_T = DataFrame(data=T, columns=T_var)
    df_Y = DataFrame(data=Y, columns=['RUL'])
    df_A = DataFrame(data=A, columns=A_var)

    # Merge all the dataframes
    df_all = pd.concat([df_W, df_Xs, df_Xv, df_T, df_Y, df_A], axis=1)

    '''
    Split dataframe into Train and Test
    Training units: 2, 5, 10, 16, 18, 20
    Test units: 11, 14, 15

    '''
    units = list(np.unique(df_A['unit']))
    units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
    units_index_test = [11.0, 14.0, 15.0]
    train_df_lst = []
    test_df_lst = []
    print("units_index_train", units_index_train)
    print("units_index_test", units_index_test)

    for idx in units_index_train:
        df_train_temp = df_all[df_all['unit'] == np.float64(idx)]
        train_df_lst.append(df_train_temp)

    df_train = pd.concat(train_df_lst)
    df_train = df_train.reset_index(drop=True)
    print(df_train)

    for idx in units_index_test:
        df_test_temp = df_all[df_all['unit'] == np.float64(idx)]
        test_df_lst.append(df_test_temp)

    df_test = pd.concat(test_df_lst)
    df_test = df_test.reset_index(drop=True)
    print(df_test)


if __name__ == '__main__':
    main()
